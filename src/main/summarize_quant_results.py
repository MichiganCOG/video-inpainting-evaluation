import argparse
import re
from collections import namedtuple

import numpy as np
import prettytable as pt
from scipy.stats import sem


TableMetricProps = namedtuple('MetricProps', ['header', 'format'])

METRIC_INFO = {
    'psnr': TableMetricProps('PSNR \u25B2', '{:.2f}'),
    'ssim': TableMetricProps('SSIM \u25B2', '{:.4f}'),
    'lpips': TableMetricProps('LPIPS \u25BC', '{:.6f}'),
    'pvcs': TableMetricProps('PVCS \u25BC', '{:.4f}'),
    'fid': TableMetricProps('FID \u25BC', '{:.2f}'),
    'vfid': TableMetricProps('VFID \u25BC', '{:.4f}'),
    'vfid_clips': TableMetricProps('VFID (clips) \u25BC', '{:.4f}'),
    'warp_error': TableMetricProps('Warp error \u25BC', '{:.6f}'),
    'warp_error_mask': TableMetricProps('Warp error (mask) \u25BC', '{:.6f}'),
    'pcons_psnr': TableMetricProps('PCons (PSNR) \u25B2', '{:.2f}'),
    'pcons_psnr_mask': TableMetricProps('PCons (PSNR, mask) \u25B2', '{:.2f}'),
    'pcons_ssim': TableMetricProps('PCons (SSIM) \u25B2', '{:.4f}'),
}


def main(result_paths, hide_list, show_list, print_tsv, no_se):
    # Determine which metrics to show
    if not show_list:
        # Show all columns by default
        show_list = [metric_key for metric_key in METRIC_INFO]
        # Remove columns if hide_list is specified
        if hide_list:
            show_list = list(filter(lambda x: x not in hide_list, show_list))

    table = pt.PrettyTable()
    table_headers = [v.header for k, v in METRIC_INFO.items() if k in show_list]
    table.field_names = ['Method'] + table_headers
    table.align['Method'] = 'l'

    for result_path in result_paths:
        add_summary(table, result_path, no_se, table_headers, show_list)

    if print_tsv:
        table.hrules = pt.NONE
        table.padding_width = 1
        table.left_padding_width = 0
        table.right_padding_width = 0
        table.vertical_char = '\t'

        ret = table.get_string()

        ret = re.sub(r'  +', '', ret)
        ret = re.sub(r'^\t', '', ret)
        ret = re.sub(r'\n\t', '\n', ret)
        ret = re.sub(r'\t\n', '\n', ret)
        ret = re.sub(r'\t$', '', ret)
    else:
        ret = table.get_string()

    print(ret)


def add_summary(table, result_path, no_se, table_headers, metric_keys):
    npz = np.load(result_path)

    # Process per-video metric values (e.g., clip values, compute mean and standard error)
    data = {}
    for metric_key in metric_keys:
        metric_values = npz.get(metric_key)
        # Clip PSNR-based metrics
        if metric_key in ['psnr', 'pcons_psnr', 'pcons_psnr_mask']:
            metric_values = np.clip(metric_values, 0, max_non_inf(metric_values))
        if metric_values.size > 1:
            data['{}_mean'.format(metric_key)] = np.mean(metric_values)
            data['{}_sem'.format(metric_key)] = sem(metric_values)
        else:
            data[metric_key] = metric_values.item()

    # Construct the strings that will make up each column in the current row
    row = [result_path]
    for header, metric_key in zip(table_headers, metric_keys):
        col_format = METRIC_INFO[metric_key].format
        mean = data.get('{}_mean'.format(metric_key), data.get(metric_key))
        sem_value = data.get('{}_sem'.format(metric_key), None)
        if sem_value is not None and not no_se:
            row_item = '{} \u00B1 {}'.format(col_format, col_format).format(mean, sem_value)
        else:
            row_item = col_format.format(mean)
        row.append(row_item)
    table.add_row(row)


def max_non_inf(a):
    return max(a[np.where(a < np.inf)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-r', '--result_paths', type=str, nargs='+', required=True,
                        help='The path to the .npz file used to store results')
    parser.add_argument('-t', '--print_tsv', action='store_true',
                        help='Flag to print table in TSV format')
    parser.add_argument('--no-se', action='store_true',
                        help='Flag to not print standard error')

    hide_show_args = parser.add_mutually_exclusive_group()
    hide_show_args.add_argument('-h', '--hide', type=str, nargs='+', dest='hide_list', help='The columns to hide')
    hide_show_args.add_argument('-s', '--show', type=str, nargs='+', dest='show_list', help='The columns to show')
    args = parser.parse_args()

    main(**vars(args))
