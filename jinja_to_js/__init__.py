# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import contextlib
import logging
import json
import re
import os
import sys

from os import path

from jinja2 import Environment, FileSystemLoader, nodes
import six


OPERANDS = {
    'eq': '===',
    'ne': '!==',
    'lt': ' < ',
    'gt': ' > ',
    'lteq': ' <= ',
    'gteq': ' >= '
}

DICT_ITER_METHODS = (
    'iteritems',
    'items',
    'values',
    'keys'
)

STATE_DEFAULT = 0
STATE_EXECUTING = 1
STATE_INTERPOLATING = 2

LOOP_HELPER_INDEX = 'index'
LOOP_HELPER_INDEX_0 = 'index0'
LOOP_HELPER_FIRST = 'first'
LOOP_HELPER_LAST = 'last'
LOOP_HELPER_LENGTH = 'length'
LOOP_HELPERS = (
    LOOP_HELPER_INDEX,
    LOOP_HELPER_INDEX_0,
    LOOP_HELPER_FIRST,
    LOOP_HELPER_LAST,
    LOOP_HELPER_LENGTH
)


def amd_format(dependencies, template_function):
    result = 'define(['
    result += ",".join('"{0}"'.format(x[0]) for x in dependencies)
    result += '], function ('
    result += ",".join(x[1] for x in dependencies)
    result += ') { return '
    result += template_function
    result += '; });'
    return result


def commonjs_format(dependencies, template_function):
    result = ''.join('var {0} = require("{1}");'.format(y, x) for x, y in dependencies)
    result += 'module.exports = {0};'.format(template_function)
    return result


def es6_format(dependencies, template_function):
    result = ''.join('import {0} from "{1}";'.format(y, x) for x, y in dependencies)
    result += 'export default {0}'.format(template_function)
    return result


JS_MODULE_FORMATS = {
    None: lambda dependencies, template_function: template_function,
    'amd': amd_format,
    'commonjs': commonjs_format,
    'es6': es6_format
}


# This string has to double all the '{' and '}' due to Python's string formatting.
# See - https://docs.python.org/2/library/string.html#formatstrings
TEMPLATE_WRAPPER = """
function {function_name}(ctx) {{
    var __result = "";
    var __tmp;
    var __runtime = jinjaToJS.runtime;
    var __filters = jinjaToJS.filters;
    var __globals = jinjaToJS.globals;
    var context = jinjaToJS.createContext(ctx);
    {template_code}
    return __result;
}}
"""


class ExtendsException(Exception):
    """
    Raised when an {% extends %} is encountered. At this point the parent template is
    loaded and all blocks defined in the current template passed to it.
    """
    pass


@contextlib.contextmanager
def option(current_kwargs, **kwargs):
    """
    Context manager for temporarily setting a keyword argument and
    then restoring it to whatever it was before.
    """

    tmp_kwargs = dict((key, current_kwargs.get(key)) for key, value in kwargs.items())
    current_kwargs.update(kwargs)
    yield
    current_kwargs.update(tmp_kwargs)


def is_method_call(node, method_name):
    """
    Returns True if `node` is a method call for `method_name`. `method_name`
    can be either a string or an iterable of strings.
    """

    if not isinstance(node, nodes.Call):
        return False

    if isinstance(node.node, nodes.Getattr):
        # e.g. foo.bar()
        method = node.node.attr

    elif isinstance(node.node, nodes.Name):
        # e.g. bar()
        method = node.node.name

    elif isinstance(node.node, nodes.Getitem):
        # e.g. foo["bar"]()
        method = node.node.arg.value

    else:
        return False

    if isinstance(method_name, (list, tuple)):
        return method in method_name

    return method == method_name


def is_loop_helper(node):
    """
    Returns True is node is a loop helper e.g. {{ loop.index }} or {{ loop.first }}
    """
    return hasattr(node, 'node') and isinstance(node.node, nodes.Name) and node.node.name == 'loop'


def temp_var_names_generator():
    x = 0
    while True:
        yield '__$%s' % x
        x += 1


def clean_path(file_path):
    # Jinja2 doesn't accept Windows filepaths (but does output them!)
    if os.name == 'nt':
        return file_path.replace(os.path.sep, '/')
    return file_path


class JinjaToJS(object):

    def __init__(self,
                 template_root,
                 template_name,
                 js_module_format=None,
                 runtime_path='jinja-to-js',
                 include_prefix='',
                 include_ext='',
                 child_blocks=None,
                 dependencies=None,
                 custom_filters=None):
        """
        Processing starts at the last child template, and parent templates are processed
        each time an extends tag is encountered.
        
        Inheritance is handled by keeping a master dict of all blocks (from all templates)
        with the name forming a key to a chain from child->parent->grandparent via the
        'super_block' attribute of each block.

        After each template is parsed by jinja, the dict of blocks is updated and
        inheritance chains extended as appropriate.
        
        Args:
            template_root (str): The path to where templates should be loaded from.
            template_name (str): The name of the template to compile (relative to `template_root`).
            js_module_format (str, optional): The JavaScript module format to use.
                                              One of ('amd', 'commonjs', 'es6')
            runtime_path (str, optional): If `js_module_format` is specified then the JavaScript
                                          runtime will be imported using the appropriate method.
                                          It defaults to assuming it will be imported from
                                          `node_modules` but you can change it using this option.
            include_prefix (str, optional): If using the `amd` module format you can use this option
                                            to add a prefix to every include path as AMD imports are
                                            generally relative to the main file, not the module
                                            importing.
            include_ext (str, optional): By default any includes will be references without an
                                         extension, as neither AMD, commonJS or ES6 require the
                                         '.js' extension. If you want to use an extension, say
                                         '.template' then set this option to a string including
                                         the leading '.'
            child_blocks (dict, optional): A dict of all the already defined templates.
            dependencies (list of tuple, optional): Used internally when handling templates that
                                                    extend other templates.
            custom_filters (list of str, optional): List of custom filters which should be allowed.
                                                    These may be filters supported by Jinja but not
                                                    supported by jinja-to-js. These filters MUST be
                                                    registered with the jinja-to-js JS runtime.
        """

        self.environment = Environment(loader=FileSystemLoader(template_root),
                                       autoescape=True,
                                       extensions=['jinja2.ext.with_', 'jinja2.ext.autoescape'])
        self.output = six.StringIO()
        self.stored_names = set()
        self.temp_var_names = temp_var_names_generator()
        self.state = STATE_DEFAULT
        self.dependencies = dependencies or []
        self._runtime_function_cache = []
        self.js_module_format = js_module_format
        self.runtime_path = runtime_path
        self.include_prefix = include_prefix
        self.include_ext = include_ext
        self.template_root = template_root
        self.template_name = clean_path(template_name)
        self.custom_filters = custom_filters or []

        # The name of the JavaScript function that will output this template. By using a named
        # function the template can call itself which is required to support recursive includes.
        self.js_function_name = 'template' + ''.join(
            x.title() for x in re.split(r'[^\w]|_', path.splitext(self.template_name)[0])
        )

        self.context_name = 'context'

        self._add_dependency(self.runtime_path, 'jinjaToJS')

        template_string, template_path, _ = self.environment.loader.get_source(
            self.environment, self.template_name
        )

        # It is assumed that this will be the absolute path to the template. It is used to work out
        # related paths for inclues.
        self.template_path = template_path

        if self.js_module_format not in JS_MODULE_FORMATS.keys():
            raise ValueError(
                'The js_module_format option must be one of: %s' % JS_MODULE_FORMATS.keys()
            )

        logging.info('Parsing template: %s', template_string)
        self.ast = self.environment.parse(template_string)

        # re-use the passed blocks if we are not the child template
        if child_blocks is None:
            # we are the first template to be processed, so create the master blocks list
            self.blocks = {}
        else:
            self.blocks = child_blocks

        local_blocks = set()
        for block in self.ast.find_all(nodes.Block):
            # check for duplicate blocks in this template
            if block.name in local_blocks:
                raise Exception(f"block {block.name!r} defined twice", block.lineno)
            local_blocks.add(block.name)

            # if not already in the block list then this is the first time a
            # block with this name has been encountered.
            if block.name not in self.blocks:
                self.blocks[block.name] = block
            else:
                # otherwise we have seen this block before, so we need to find the last
                # super_block and add the block from this template to the end of the list
                last_super = self.blocks.get(block.name)
                while hasattr(last_super, 'super_block'):
                    last_super = last_super.super_block
                last_super.super_block = block

        self.current_block = None
        
        try:
            for node in self.ast.body:
                self._process_node(node)
        except ExtendsException:
            pass

    def get_output(self):
        """
        Returns the generated JavaScript code.

        Returns:
            str
        """
        # generate the JS function string
        template_function = TEMPLATE_WRAPPER.format(
            function_name=self.js_function_name,
            template_code=self.output.getvalue()
        ).strip()

        # get the correct module format template
        module_format = JS_MODULE_FORMATS[self.js_module_format]

        # generate the module code
        return module_format(self.dependencies, template_function)

    def _get_depencency_var_name(self, dependency):
        """
        Returns the variable name assigned to the given dependency or None if the dependency has
        not yet been registered.

        Args:
            dependency (str): Thet dependency that needs to be imported.

        Returns:
            str or None
        """
        for dep_path, var_name in self.dependencies:
            if dep_path == dependency:
                return var_name

    def _add_dependency(self, dependency, var_name=None):
        """
        Adds the given dependency and returns the variable name to use to access it. If `var_name`
        is not given then a random one will be created.

        Args:
            dependency (str):
            var_name (str, optional):

        Returns:
            str
        """
        dependency = clean_path(dependency)
        if var_name is None:
            var_name = next(self.temp_var_names)
        # Don't add duplicate dependencies
        if (dependency, var_name) not in self.dependencies:
            self.dependencies.append((dependency, var_name))
        return var_name

    def _process_node(self, node, **kwargs):
        node_name = node.__class__.__name__.lower()
        handler = getattr(self, '_process_' + node_name, None)
        if callable(handler):
            handler(node, **kwargs)
        else:
            raise Exception('Unknown node %s' % node)

    def _process_extends(self, node, **kwargs):
        """
        Processes an extends block e.g. `{% extends "some/template.jinja" %}`
        """
        logging.debug('Extends parent template: %s', node.template.value)
        # load the parent template
        parent_template = JinjaToJS(template_root=self.template_root,
                                    template_name=node.template.value,
                                    js_module_format=self.js_module_format,
                                    runtime_path=self.runtime_path,
                                    include_prefix=self.include_prefix,
                                    include_ext=self.include_ext,
                                    child_blocks=self.blocks,
                                    dependencies=self.dependencies,
                                    custom_filters=self.custom_filters)

        # add the parent templates output to the current output
        self.output.write(parent_template.output.getvalue())

        # Raise an exception so we stop parsing this template
        raise ExtendsException

    def _process_block(self, block, **kwargs):
        """
        Processes a block e.g. `{% block my_block %}{% endblock %}`
        """
        logging.info('Processing block %s in %s', block.name, self.template_name)
        
        # only process if we are an explicit parent call, or the last child with this name
        if kwargs.get('explicit_super', False):
            logging.debug('Processing block %s content as explicit super() call', block.name)
        elif block == self.blocks.get(block.name):
            logging.debug('Processing block %s content as last child', block.name)
        else:
            logging.debug('Calling last child of block %s', block.name)
            self._process_node(self.blocks.get(block.name), **kwargs)
            return

        outer_block = self.current_block
        self.current_block = block

        # process the block 
        # if this block # calls super() it will be handled by `_process_call`
        with option(kwargs, explicit_super=False):
            for node in block.body:
                self._process_node(node, **kwargs)

        logging.info('End of block %s in %s', block.name, self.template_name)
        self.current_block = outer_block

    def _process_output(self, node, **kwargs):
        """
        Processes an output node, which will contain things like `Name` and `TemplateData` nodes.
        """
        for n in node.nodes:
            self._process_node(n, **kwargs)

    def _process_templatedata(self, node, **_):
        """
        Processes a `TemplateData` node, this is just a bit of as-is text
        to be written to the output.
        """

        # escape double quotes
        value = re.sub('"', r'\\"', node.data)

        # escape new lines
        value = re.sub('\n', r'\\n', value)

        # append value to the result
        self.output.write('__result += "' + value + '";')

    def _process_name(self, node, **kwargs):
        """
        Processes a `Name` node. Some examples of `Name` nodes:
            {{ foo }} -> 'foo' is a Name
            {% if foo }} -> 'foo' is a Name
        """

        with self._interpolation():
            with self._python_bool_wrapper(**kwargs):

                if node.name not in self.stored_names and node.ctx != 'store':
                    self.output.write(self.context_name)
                    self.output.write('.')

                if node.ctx == 'store':
                    self.stored_names.add(node.name)

                self.output.write(node.name)

    def _process_getattr(self, node, **kwargs):
        """
        Processes a `GetAttr` node. e.g. {{ foo.bar }}
        """

        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                if is_loop_helper(node):
                    self._process_loop_helper(node, **new_kwargs)
                else:
                    self._process_node(node.node, **new_kwargs)
                    self.output.write('.')
                    self.output.write(node.attr)

    def _process_getitem(self, node, **kwargs):
        """
        Processes a `GetItem` node e.g. {{ foo["bar"] }}
        """

        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)

                if isinstance(node.arg, nodes.Slice):
                    self.output.write('.slice(')

                    if node.arg.step is not None:
                        raise Exception('The step argument is not supported when slicing.')

                    if node.arg.start is None:
                        self.output.write('0')
                    else:
                        self._process_node(node.arg.start, **new_kwargs)

                    if node.arg.stop is None:
                        self.output.write(')')
                    else:
                        self.output.write(',')
                        self._process_node(node.arg.stop, **new_kwargs)
                        self.output.write(')')
                else:
                    self.output.write('[')
                    self._process_node(node.arg, **new_kwargs)
                    self.output.write(']')

    def _process_for(self, node, **kwargs):
        """
        Processes a for loop. e.g.
            {% for number in numbers %}
                {{ number }}
            {% endfor %}
            {% for key, value in somemap.items() %}
                {{ key }} -> {{ value }}
            {% %}
        """

        # since a for loop can introduce new names into the context
        # we need to remember the ones that existed outside the loop
        previous_stored_names = self.stored_names.copy()

        with self._execution():
            self.output.write('__runtime.each(')

            if is_method_call(node.iter, dict.keys.__name__):
                self.output.write('Object.keys(')

            self._process_node(node.iter, **kwargs)

            if is_method_call(node.iter, dict.keys.__name__):
                self.output.write(')')

            self.output.write(',')
            self.output.write('function')
            self.output.write('(')

            # javascript iterations put the value first, then the key
            if isinstance(node.target, nodes.Tuple):
                if len(node.target.items) > 2:
                    raise Exception('De-structuring more than 2 items is not supported.')

                for i, item in enumerate(reversed(node.target.items)):
                    self._process_node(item, **kwargs)
                    if i < len(node.target.items) - 1:
                        self.output.write(',')
            else:
                self._process_node(node.target, **kwargs)

            self.output.write(')')
            self.output.write('{')

            if node.test:
                self.output.write('if (!(')
                self._process_node(node.test, **kwargs)
                self.output.write(')) { return; }')

        assigns = node.target.items if isinstance(node.target, nodes.Tuple) else [node.target]

        with self._scoped_variables(assigns, **kwargs):
            for n in node.body:
                self._process_node(n, **kwargs)

        with self._execution():
            self.output.write('}')
            self.output.write(')')
            self.output.write(';')

        # restore the stored names
        self.stored_names = previous_stored_names

    def _process_if(self, node, execute_end=None, **kwargs):
        """
        Processes an if block e.g. `{% if foo %} do something {% endif %}`
        """

        with self._execution():
            self.output.write('if')
            self.output.write('(')

            with option(kwargs, use_python_bool_wrapper=True):
                self._process_node(node.test, **kwargs)

            self.output.write(')')
            self.output.write('{')

        # We accept an `execute_end` function as a keyword argument as this function is
        # recursive in the case of something like if-elif-elif-else. In these cases this
        # invocation of this function may have to close execution opened by a previous
        # invocation of this function.
        if execute_end:
            execute_end()

        # body
        for n in node.body:
            self._process_node(n, **kwargs)

        if not node.else_ and not node.elif_:
            # no else - just close the if
            with self._execution():
                self.output.write('}')

        else:
            # either an else or an elif
            with self._execution() as execute_end:
                self.output.write('}')
                self.output.write(' else ')

                # check for elif
                for n in node.elif_:
                    self._process_node(n, execute_end=execute_end, **kwargs)

                if node.elif_ and node.else_:
                    self.output.write(' else ')

                # open up the body
                self.output.write('{')

            # process the body of the else
            for n in node.else_:
                self._process_node(n, **kwargs)

            # close the body
            with self._execution():
                self.output.write('}')

    def _process_condexpr(self, node, **kwargs):
        with self._interpolation():
            self.output.write('(')

            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.test, **new_kwargs)

            self.output.write(' ? ')
            self._process_node(node.expr1, **kwargs)
            self.output.write(' : ')
            self._process_node(node.expr2, **kwargs)
            self.output.write(')')

    def _process_not(self, node, **kwargs):
        self.output.write('!')

        with self._python_bool_wrapper(**kwargs) as new_kwargs:
            self._process_node(node.node, **new_kwargs)

    def _process_or(self, node, **kwargs):
        self._process_node(node.left, **kwargs)
        self.output.write(' || ')
        self._process_node(node.right, **kwargs)

    def _process_and(self, node, **kwargs):
        self._process_node(node.left, **kwargs)
        self.output.write(' && ')
        self._process_node(node.right, **kwargs)

    def _process_tuple(self, node, **kwargs):
        self.output.write('[')
        for i, item in enumerate(node.items):
            self._process_node(item, **kwargs)
            if i < len(node.items) - 1:
                self.output.write(',')
        self.output.write(']')

    def _process_call(self, node, **kwargs):
        if is_method_call(node, DICT_ITER_METHODS):
            # special case for dict methods
            self._process_node(node.node.node, **kwargs)

        elif is_method_call(node, 'super'):
            # special case for the super() method which is available inside blocks
            if not (hasattr(self.current_block, 'super_block') and self.current_block.super_block):
                if self.current_block:
                    current_block_name = self.current_block.name
                else:
                    current_block_name = 'None'
                raise Exception('super() called outside of a block with a parent. (in block {})'.format(current_block_name))
            with option(kwargs, explicit_super=True):
                # TODO: handle chained super.super()
                super_block = self.current_block.super_block

                levels = 1
                subnode = node
                while isinstance(subnode.node, nodes.Getattr):
                    # e.g. super.super()
                    levels += 1
                    subnode = subnode.node
                    super_block = super_block.super_block

                logging.debug('Process call to %s x super() from block %s in %s', levels, self.current_block.name, self.template_name)
                self._process_node(super_block, **kwargs)

        else:
            # just a normal function call on a context variable
            with self._interpolation():
                with self._python_bool_wrapper(**kwargs) as new_kwargs:
                    self._process_node(node.node, **new_kwargs)
                    self.output.write('(')
                    self._process_args(node, **new_kwargs)
                    self.output.write(')')

                    # only output the semi-colon if we are not interpolating
                    if self.state != STATE_INTERPOLATING:
                        self.output.write('')

    def _process_filter(self, node, **kwargs):
        method_name = getattr(self, '_process_filter_%s' % node.name, None)
        if callable(method_name):
            method_name(node, **kwargs)
        elif node.name in self.custom_filters:
            with self._interpolation(safe=True):
                with self._python_bool_wrapper(**kwargs) as new_kwargs:
                    self.output.write('__filters.%s(' % node.name)
                    self._process_node(node.node, **new_kwargs)
                    if getattr(node, 'args', None):
                        self.output.write(',')
                        self._process_args(node, **new_kwargs)
                    self.output.write(')')
        else:
            raise Exception('Unsupported filter: %s' % node.name)

    def _process_filter_safe(self, node, **kwargs):
        with self._interpolation(safe=True):
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)

    def _process_filter_capitalize(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.capitalize(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_abs(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('Math.abs(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_attr(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self._process_node(node.node, **new_kwargs)
                self.output.write('[')
                self._process_node(node.args[0], **new_kwargs)
                self.output.write(']')

    def _process_filter_batch(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.batch(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_default(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.default(')
                self._process_node(node.node, **new_kwargs)
                if node.args:
                    self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_first(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.first(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_int(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.int(')
                self._process_node(node.node, **new_kwargs)
                if node.args:
                    self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_last(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.last(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_length(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.size(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_lower(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(' + "").toLowerCase()')

    def _process_filter_slice(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.slice(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_filter_title(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.title(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(')')

    def _process_filter_trim(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(' + "").trim()')

    def _process_filter_upper(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(' + "").toUpperCase()')

    def _process_filter_truncate(self, node, **kwargs):
        with self._interpolation():
            with self._python_bool_wrapper(**kwargs) as new_kwargs:
                self.output.write('__filters.truncate(')
                self._process_node(node.node, **new_kwargs)
                self.output.write(',')
                self._process_args(node, **new_kwargs)
                self.output.write(')')

    def _process_assign(self, node, **kwargs):
        with self._execution():
            self.output.write('var ')
            self._process_node(node.target, **kwargs)
            self.output.write(' = ')
            self._process_node(node.node, **kwargs)
            self.output.write(';')

    def _process_with(self, node, **kwargs):

        # keep a copy of the stored names before the scope
        previous_stored_names = self.stored_names.copy()

        # assigns in the with tag
        # e.g. {% with var = "something %}
        assigns_in_tag = [nodes.Assign(t, v) for t, v in zip(node.targets, node.values)]

        # assigns in the with body
        # e.g. {% set name = 'John' %}
        assigns_in_body = [x for x in node.body if isinstance(x, nodes.Assign)]

        # remove assigns from the body
        node.body = [x for x in node.body if not isinstance(x, nodes.Assign)]

        # get a list of all the assigns in this with block
        # both on the tag, and within the body of the block
        all_assigns = assigns_in_tag + assigns_in_body

        with self._execution():
            self.output.write('(function () {')

        with self._scoped_variables(all_assigns, **kwargs):
            for node in node.body:
                self._process_node(node, **kwargs)

        with self._execution():
            self.output.write('})();')

        # restore previous stored names
        self.stored_names = previous_stored_names

    def _process_compare(self, node, **kwargs):

        if len(node.ops) > 1:
            raise Exception('Multiple operands are not supported.')

        operand = node.ops[0]
        is_equality = operand.op in ('eq', 'ne')
        left_hand_is_const = isinstance(node.expr, nodes.Const)
        right_hand_is_const = isinstance(operand.expr, nodes.Const)

        # If the operand is equality and neither the left or right hand side are constants then we
        # will need to use the JavaScript deep equals function. Ideally we want to avoid using this
        # as it is quite a big function.
        use_is_equal_function = is_equality and not (left_hand_is_const or right_hand_is_const)

        with option(kwargs, use_python_bool_wrapper=False):

            if use_is_equal_function:
                if operand.op == 'ne':
                    self.output.write('!')
                self.output.write('__runtime.isEqual(')

            self._process_node(node.expr, **kwargs)

            if use_is_equal_function:
                self.output.write(',')
            else:
                self.output.write(OPERANDS.get(operand.op))

            self._process_node(operand.expr, **kwargs)

            if use_is_equal_function:
                self.output.write(')')

    def _process_operand(self, node, **kwargs):
        self.output.write(OPERANDS.get(node.op))
        self._process_node(node.expr, **kwargs)

    def _process_const(self, node, **_):
        with self._interpolation():
            self.output.write(json.dumps(node.value))

    def _process_nonetype(self, node, **_):
        with self._interpolation():
            self.output.write('null')

    def _process_neg(self, node, **kwargs):
        with self._interpolation():
            self.output.write('-')
            self._process_node(node.node, **kwargs)

    def _process_list(self, node, **kwargs):
        self.output.write('[')
        for i, item in enumerate(node.items):
            self._process_node(item, **kwargs)
            if i < len(node.items) - 1:
                self.output.write(',')
        self.output.write(']')

    def _process_test(self, node, **kwargs):
        with option(kwargs, use_python_bool_wrapper=False):
            method_name = getattr(self, '_process_test_%s' % node.name, None)
            if callable(method_name):
                method_name(node, **kwargs)
            else:
                raise Exception('Unsupported test: %s' % node.name)

    def _process_test_defined(self, node, **kwargs):
        self.output.write('(typeof ')
        self._process_node(node.node, **kwargs)
        self.output.write(' !== "undefined")')

    def _process_test_undefined(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write(' === undefined')

    def _process_test_callable(self, node, **kwargs):
        self.output.write('__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "Function"')

    def _process_test_divisibleby(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write(' % ')
        self._process_node(node.args[0], **kwargs)
        self.output.write(' === 0')

    def _process_test_even(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write(' % 2 === 0')

    def _process_test_odd(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write(' % 2 === 1')

    def _process_test_none(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write(' === null')

    def _process_test_upper(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write('.toUpperCase() === ')
        self._process_node(node.node, **kwargs)

    def _process_test_lower(self, node, **kwargs):
        self._process_node(node.node, **kwargs)
        self.output.write('.toLowerCase() === ')
        self._process_node(node.node, **kwargs)

    def _process_test_string(self, node, **kwargs):
        self.output.write('__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "String"')

    def _process_test_mapping(self, node, **kwargs):
        self.output.write('__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "Object"')

    def _process_test_number(self, node, **kwargs):
        self.output.write('(__runtime.type(')
        self._process_node(node.node, **kwargs)
        self.output.write(') === "Number" && !isNaN(')
        self._process_node(node.node, **kwargs)
        self.output.write('))')

    def _process_include(self, node, **kwargs):
        with self._interpolation(safe=True):
            include_path = node.template.value

            if include_path == self.template_name:
                # template is including itself
                include_var_name = self.js_function_name
            else:
                if self.include_prefix:
                    include_path = self.include_prefix + node.template.value
                elif self.js_module_format in ('es6', 'commonjs',) and self.template_name:
                    _, absolute_include_path, _ = self.environment.loader.get_source(
                        self.environment, node.template.value
                    )
                    include_path = os.path.relpath(
                        absolute_include_path, os.path.dirname(self.template_path)
                    )
                    if not include_path.startswith('.'):
                        include_path = './' + include_path

                include_path = clean_path(include_path)

                include_path = path.splitext(include_path)[0] + self.include_ext
                include_var_name = self._get_depencency_var_name(include_path)

                if not include_var_name:
                    include_var_name = self._add_dependency(include_path)

            if self.js_module_format is None:
                self.output.write('jinjaToJS.include("')
                self.output.write(include_path)
                self.output.write('");')
            else:
                self.output.write(include_var_name)

            self.output.write('(')
            self.output.write(self.context_name)
            self.output.write(')')

    def _process_add(self, node, **kwargs):
        self._process_math(node, math_operator=' + ', **kwargs)

    def _process_sub(self, node, **kwargs):
        self._process_math(node, math_operator=' - ', **kwargs)

    def _process_div(self, node, **kwargs):
        self._process_math(node, math_operator=' / ', **kwargs)

    def _process_floordiv(self, node, **kwargs):
        self._process_math(node, math_operator=' / ', function='Math.floor', **kwargs)

    def _process_mul(self, node, **kwargs):
        self._process_math(node, math_operator=' * ', **kwargs)

    def _process_mod(self, node, **kwargs):
        self._process_math(node, math_operator=' % ', **kwargs)

    def _process_math(self, node, math_operator=None, function=None, **kwargs):
        """
        Processes a math node e.g. `Div`, `Sub`, `Add`, `Mul` etc...
        If `function` is provided the expression is wrapped in a call to that function.
        """

        with self._interpolation():
            if function:
                self.output.write(function)
                self.output.write('(')

            self._process_node(node.left, **kwargs)
            self.output.write(math_operator)
            self._process_node(node.right, **kwargs)

            if function:
                self.output.write(')')

    def _process_loop_helper(self, node, **kwargs):
        """
        Processes a loop helper e.g. {{ loop.first }} or {{ loop.index }}
        """

        if node.attr == LOOP_HELPER_INDEX:
            self.output.write('(arguments[1] + 1)')
        elif node.attr == LOOP_HELPER_INDEX_0:
            self.output.write('arguments[1]')
        elif node.attr == LOOP_HELPER_FIRST:
            self.output.write('(arguments[1] == 0)')
        elif node.attr == LOOP_HELPER_LAST:
            self.output.write('(arguments[1] == arguments[2].length - 1)')
        elif node.attr == LOOP_HELPER_LENGTH:
            self.output.write('arguments[2].length')

    def _process_args(self, node, **kwargs):
        args = getattr(node, 'args', None)
        if not args:
            return
        for i, item in enumerate(args):
            self._process_node(item, **kwargs)
            if i < len(node.args) - 1:
                self.output.write(',')

    @contextlib.contextmanager
    def _execution(self):
        """
        Context manager for executing some JavaScript inside a template.
        """

        did_start_executing = False

        if self.state == STATE_DEFAULT:
            did_start_executing = True
            self.state = STATE_EXECUTING

        def close():
            if did_start_executing and self.state == STATE_EXECUTING:
                self.state = STATE_DEFAULT

        yield close
        close()

    @contextlib.contextmanager
    def _interpolation(self, safe=False):

        did_start_interpolating = False

        if self.state == STATE_DEFAULT:
            did_start_interpolating = True
            self.output.write('__result += "" + ')
            if safe is not True:
                self.output.write('__runtime.escape')
            self.output.write('((__tmp = (')
            self.state = STATE_INTERPOLATING

        def close():
            if did_start_interpolating and self.state == STATE_INTERPOLATING:
                self.output.write(')) == null ? "" : __tmp);')
                self.state = STATE_DEFAULT

        yield close
        close()

    @contextlib.contextmanager
    def _scoped_variables(self, nodes_list, **kwargs):
        """
        Context manager for creating scoped variables defined by the nodes in `nodes_list`.
        These variables will be added to the context, and when the context manager exits the
        context object will be restored to it's previous state.
        """

        tmp_vars = []
        for node in nodes_list:

            is_assign_node = isinstance(node, nodes.Assign)
            name = node.target.name if is_assign_node else node.name

            # create a temp variable name
            tmp_var = next(self.temp_var_names)

            # save previous context value
            with self._execution():

                # save the current value of this name
                self.output.write('var %s = %s.%s;' % (tmp_var, self.context_name, name))

                # add new value to context
                self.output.write('%s.%s = ' % (self.context_name, name))

                if is_assign_node:
                    self._process_node(node.node, **kwargs)
                else:
                    self.output.write(node.name)

                self.output.write(';')

            tmp_vars.append((tmp_var, name))

        yield

        # restore context
        for tmp_var, name in tmp_vars:
            with self._execution():
                self.output.write('%s.%s = %s;' % (self.context_name, name, tmp_var))

    @contextlib.contextmanager
    def _python_bool_wrapper(self, **kwargs):

        use_python_bool_wrapper = kwargs.get('use_python_bool_wrapper')

        if use_python_bool_wrapper:
            self.output.write('__runtime.boolean(')

        with option(kwargs, use_python_bool_wrapper=False):
            yield kwargs

        if use_python_bool_wrapper:
            self.output.write(')')
