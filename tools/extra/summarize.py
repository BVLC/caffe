#!/usr/bin/env python

"""Net summarization tool.

This tool summarizes the structure of a net in a concise but comprehensive
tabular listing, taking a prototxt file as input.

Use this tool to check at a glance that the computation you've specified is the
computation you expect.
"""

from caffe.proto import caffe_pb2
from google import protobuf
import re
import argparse

# ANSI codes for coloring blobs (used cyclically)
COLORS = ['92', '93', '94', '95', '97', '96', '42', '43;30', '100',
          '444', '103;30', '107;30']
DISCONNECTED_COLOR = '41'

def read_net(filename, phase=None, stages=None, level=None):
    net = caffe_pb2.NetParameter()
    with open(filename) as f:
        protobuf.text_format.Parse(f.read(), net)

    if phase is not None or stages is not None or level is not None:
        # Create the rule
        rule = caffe_pb2.NetStateRule()
        if phase.lower() == 'train':
            protobuf.text_format.Merge('phase: TRAIN', rule)
        elif phase.lower() == 'test':
            protobuf.text_format.Merge('phase: TEST', rule)
        if stages:
            for stage in stages:
                protobuf.text_format.Merge('stage: "%s"' % stage, rule)
        if level is not None:
            protobuf.text_format.Merge('level: %d' % level, rule)

        print '>>> NetStateRule'
        print protobuf.text_format.MessageToString(rule)

        # Filter by the rule
        layers = []
        for layer in net.layer:
            if layer_meets_rule(layer, rule):
                layers.append(layer)
        return layers

    else:
        return net.layer


def layer_meets_rule(layer, state):
    """
    Returns True if this layer will be included in the given NetStateRule
    Logic copied from Net::FilterNet()
    """
    # If no include rules are specified, the layer is included by default and
    # only excluded if it meets one of the exclude rules.
    layer_included = len(layer.include) == 0

    for exclude_rule in layer.exclude:
        if state_meets_rule(state, exclude_rule):
            layer_included = False
            break

    for include_rule in layer.include:
        if state_meets_rule(state, include_rule):
            layer_included = True
            break

    return layer_included


def state_meets_rule(state, rule):
    """
    Returns True if the given state meets the given NetStateRule
    Logic copied from Net::StateMeetsRule()
    """
    if rule.HasField('phase'):
        if rule.phase != state.phase:
            return False

    if rule.HasField('min_level'):
        if state.level < rule.min_level:
            return False

    if rule.HasField('max_level'):
        if state.level > rule.max_level:
            return False

    # The state must contain ALL of the rule's stages
    for stage in rule.stage:
        if stage not in state.stage:
            return False

    # The state must contain NONE of the rule's not_stages
    for stage in rule.not_stage:
        if stage in state.stage:
            return False

    return True


def format_param(param):
    out = []
    if len(param.name) > 0:
        out.append(param.name)
    if param.lr_mult != 1:
        out.append('x{}'.format(param.lr_mult))
    if param.decay_mult != 1:
        out.append('Dx{}'.format(param.decay_mult))
    return ' '.join(out)

def printed_len(s):
    return len(re.sub(r'\033\[[\d;]+m', '', s))

def print_table(table, max_width):
    """Print a simple nicely-aligned table.

    table must be a list of (equal-length) lists. Columns are space-separated,
    and as narrow as possible, but no wider than max_width. Text may overflow
    columns; note that unlike string.format, this will not affect subsequent
    columns, if possible."""

    max_widths = [max_width] * len(table[0])
    column_widths = [max(printed_len(row[j]) + 1 for row in table)
                     for j in range(len(table[0]))]
    column_widths = [min(w, max_w) for w, max_w in zip(column_widths, max_widths)]

    for row in table:
        row_str = ''
        right_col = 0
        for cell, width in zip(row, column_widths):
            right_col += width
            row_str += cell + ' '
            row_str += ' ' * max(right_col - printed_len(row_str), 0)
        print row_str

def summarize_net(layers):
    disconnected_tops = set()
    for lr in layers:
        disconnected_tops |= set(lr.top)
        disconnected_tops -= set(lr.bottom)

    table = []
    colors = {}
    for lr in layers:
        tops = []
        for ind, top in enumerate(lr.top):
            color = colors.setdefault(top, COLORS[len(colors) % len(COLORS)])
            if top in disconnected_tops:
                top = '\033[1;4m' + top
            if len(lr.loss_weight) > 0:
                top = '{} * {}'.format(lr.loss_weight[ind], top)
            tops.append('\033[{}m{}\033[0m'.format(color, top))
        top_str = ', '.join(tops)

        bottoms = []
        for bottom in lr.bottom:
            color = colors.get(bottom, DISCONNECTED_COLOR)
            bottoms.append('\033[{}m{}\033[0m'.format(color, bottom))
        bottom_str = ', '.join(bottoms)

        if lr.type == 'Python':
            type_str = lr.python_param.module + '.' + lr.python_param.layer
        else:
            type_str = lr.type

        # Summarize conv/pool parameters.
        # TODO support rectangular/ND parameters
        conv_param = lr.convolution_param
        if (lr.type in ['Convolution', 'Deconvolution']
                and len(conv_param.kernel_size) == 1):
            arg_str = str(conv_param.kernel_size[0])
            if len(conv_param.stride) > 0 and conv_param.stride[0] != 1:
                arg_str += '/' + str(conv_param.stride[0])
            if len(conv_param.pad) > 0 and conv_param.pad[0] != 0:
                arg_str += '+' + str(conv_param.pad[0])
            arg_str += ' ' + str(conv_param.num_output)
            if conv_param.group != 1:
                arg_str += '/' + str(conv_param.group)
        elif lr.type == 'Pooling':
            arg_str = str(lr.pooling_param.kernel_size)
            if lr.pooling_param.stride != 1:
                arg_str += '/' + str(lr.pooling_param.stride)
            if lr.pooling_param.pad != 0:
                arg_str += '+' + str(lr.pooling_param.pad)
        else:
            arg_str = ''

        if len(lr.param) > 0:
            param_strs = map(format_param, lr.param)
            if max(map(len, param_strs)) > 0:
                param_str = '({})'.format(', '.join(param_strs))
            else:
                param_str = ''
        else:
            param_str = ''

        table.append([lr.name, type_str, param_str, bottom_str, '->', top_str,
                      arg_str])
    return table

def main():
    parser = argparse.ArgumentParser(description="Print a concise summary of net computation.")
    parser.add_argument('filename', help='net prototxt file to summarize')
    parser.add_argument('-w', '--max-width', help='maximum field width',
            type=int, default=30)
    parser.add_argument('--phase', help='NetState.phase')
    parser.add_argument('--stage', help='NetState.stage',
                        nargs='*', dest='stages')
    parser.add_argument('--level', type=int, help='NetState.level')
    args = parser.parse_args()

    layers = read_net(args.filename, phase=args.phase, stages=args.stages, level=args.level)
    table = summarize_net(layers)
    print_table(table, max_width=args.max_width)

if __name__ == '__main__':
    main()
