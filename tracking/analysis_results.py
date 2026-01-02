import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist
# 'lasot' 'lasot_extension_subset'
trackers = []
dataset_name = 'nfs'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
# """FERMT"""
# trackers.extend(trackerlist(name='fermt', parameter_name='vit_tiny_ep300', dataset_name=dataset_name,
#                             save_name='fermt-291', run_ids=None, display_name='fermt256'))

# dataset = get_dataset(dataset_name)
# # dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# # plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
# #              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# # print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
# 'lasot' 'lasot_extension_subset'
dataset_name = 'lasot_extension_subset'
# trackers.extend(trackerlist(name='fermt', parameter_name='all_295', dataset_name=dataset_name,
#                             save_name='fermt', run_ids=None, display_name='fermt')) # analysis 150th epoch
# dataset = get_dataset(dataset_name)
# print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
# # '''FERMT'''
for epoch in range(291,301):
    trackers.extend(trackerlist(name='fermt', parameter_name='all_'+str(epoch), dataset_name=dataset_name,
                            save_name='fermt', run_ids=None, display_name='fermt-'+str(epoch))) # analysis 150th epoch
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('success', 'norm_prec', 'prec'))
