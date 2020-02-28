# import logging

# from climpred.prediction import compute_hindcast

# def test_log_compute_hindcast_alignment_same_init(
#     hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
# ):
#     """Checks that 'same_init' alignment has identical inits over all leads."""
#     with caplog.at_level(logging.INFO):
#         compute_hindcast(hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime)
#         for i, record in enumerate(caplog.record_tuples):
#             print(record)
#             lead = hind_ds_initialized_1d_cftime.isel(lead=i) \
#                    .lead.values.astype('int')
#             assert f'at lead={str(lead).zfill(2)}' in record[2]

#             first_year_per_lead = (
#                 hind_ds_initialized_1d_cftime.init.min().dt.year.values
#             )
#             assert f'{first_year_per_lead}' in record[2]
#             last_year_per_lead = (
#                 hind_ds_initialized_1d_cftime.init.max().dt.year.values
#                 - hind_ds_initialized_1d_cftime.lead.size
#             )
#             assert f'{last_year_per_lead}' in record[2]


# def test_log_compute_hindcast_alignment_same_init_actual_years(
#     hind_ds_initialized_1d_cftime, reconstruction_ds_1d_cftime, caplog
# ):
#     """
#     Checks that logging in compute hindcast works where we check on actual hard coded
#     numbers based on pre-defined datasets from `load_dataset`:

#     data:
#     - hind: hind_ds_initialized_1d: inits: 1955-2017, leads: 1-10
#     - verif: reconstruction_ds_1d: time: 1954-2015

#     expected verification time logged for keyword `alignment`: and pattern:
#     - inits: 1956-2006, 1957-2007, ... , 1965-2015 : first_common_init_verif+lead -
#     - verif: 1965-2006 : max(first_init+max_lead, verif_time)-min(last_init-max_lead)
#     - maximize: 1956-2015, 1957-2015, ... , 1965-2015: maximizing
#     """
#     with caplog.at_level(logging.INFO):
#         compute_hindcast(hind_ds_initialized_1d, reconstruction_ds_1d)
#         # check for each record
#         for i, record in enumerate(caplog.record_tuples):
#             print(record)
#             lead = hind_ds_initialized_1d.isel(lead=i).lead.values.astype('int')
#             assert f'at lead={str(lead).zfill(2)}' in record[2]
#             elif alignment == 'init':
#                 # check that verification length is always the same and init stay the
#                 # same, therefore verification time should monotonically decrease for
#                 # monotonic lead increases
#                 # const_verif_length = xx
#                 first_year_per_lead = (
#                     hind_ds_initialized_1d.init.min().dt.year.values + lead
#                 )
#                 assert f'{first_year_per_lead}' in record[2]
#                 last_year_per_lead = (
#                     hind_ds_initialized_1d.init.max().dt.year.values
#                     + lead
#                     - hind_ds_initialized_1d.lead.size
#                 )
#                 assert f'{last_year_per_lead}' in record[2]
#                 # check always same time length verified
#                 # assert const_verif_length == last_year_per_lead - \
#                   first_year_per_lead
