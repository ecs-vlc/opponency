import numpy as np
import pandas as pd


def _load(frame, cell_type, mode, groupby='n_bn'):
    frame = frame.drop(['max_params', 'max', 'min_params', 'min', 'spontaneous_rate', 'n_ch'], axis=1) \
        .groupby(['layer', 'd_vvs', 'n_bn', 'rep', 'class']).count().reset_index()
    frame = frame.pivot_table(values='cell', index=['layer', 'd_vvs', 'n_bn', 'rep'], columns=['class'], aggfunc=np.sum)
    frame = frame.fillna(0.0).reset_index()

    if mode + ' opponent' not in frame:
        frame[mode + ' opponent'] = 0.0
    if mode + ' non-opponent' not in frame:
        frame[mode + ' non-opponent'] = 0.0
    if mode + ' unresponsive' not in frame:
        frame[mode + ' unresponsive'] = 0.0

    total = (
            frame.groupby(['layer', groupby, 'rep'])[mode + ' opponent'].sum() +
            frame.groupby(['layer', groupby, 'rep'])[mode + ' non-opponent'].sum() +
            frame.groupby(['layer', groupby, 'rep'])[mode + ' unresponsive'].sum())

    opps = (frame.groupby(['layer', groupby, 'rep'])[cell_type].sum() / total) \
        .to_frame(name='rel_amount').reset_index()

    mopps = opps.groupby(['layer', groupby]).mean().reset_index() \
        .rename(index=str, columns={'rel_amount': 'mean_rel_amount'})
    sopps = opps.groupby(['layer', groupby]).std().reset_index() \
        .rename(index=str, columns={'rel_amount': 'std_rel_amount'})

    opps = pd.concat([mopps, sopps], axis=1, sort=False).drop_duplicates()
    opps = opps.loc[:, ~opps.columns.duplicated()]

    return opps


def spectral(frame, cell_type, groupby='n_bn'):
    return _load(frame, cell_type, 'spectrally', groupby=groupby)


def spatial(frame, cell_type, groupby='n_bn'):
    return _load(frame, cell_type, 'spatially', groupby=groupby)


def double(spectral, spatial, groupby='n_bn'):
    spatial = spatial.rename(index=str, columns={'class': 'spatial_class'}).sort_values(
        ['layer', 'cell', 'n_bn', 'd_vvs', 'rep'])
    spectral = spectral.rename(index=str, columns={'class': 'spectral_class'}).sort_values(
        ['layer', 'cell', 'n_bn', 'd_vvs', 'rep'])

    spatial.reset_index(drop=True, inplace=True)
    spectral.reset_index(drop=True, inplace=True)

    spatial = spatial.rename(index=str, columns={'class': 'spatial_class'})
    spectral = spectral.rename(index=str, columns={'class': 'spectral_class'})

    double = pd.concat([spatial, spectral], axis=1, join='outer')
    idx = pd.DataFrame({'spatial': double['spatial_class'] == 'spatially opponent',
                        'spectral': double['spectral_class'] == 'spectrally opponent'}).all(axis=1)

    frame = pd.concat([double, idx], axis=1, join='outer').rename(index=str, columns={0: 'double_opponent'})

    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = frame.drop(
        ['max_params', 'max', 'min_params', 'min', 'spontaneous_rate', 'n_ch', 'spatial_class', 'spectral_class'],
        axis=1).groupby(['layer', 'd_vvs', 'n_bn', 'rep', 'double_opponent']).count().reset_index()

    frame = frame.pivot_table(values='cell', index=['layer', 'd_vvs', 'n_bn', 'rep'], columns=['double_opponent'],
                              aggfunc=np.sum)
    frame = frame.fillna(0.0).reset_index()

    total = (frame.groupby(['layer', groupby, 'rep'])[False].sum() + frame.groupby(['layer', groupby, 'rep'])[True].sum())

    opps = (frame.groupby(['layer', groupby, 'rep'])[True].sum() / total).to_frame(name='rel_amount')

    mopps = opps.groupby(['layer', groupby]).mean().reset_index().rename(index=str, columns={'rel_amount': 'mean_rel_amount'})
    sopps = opps.groupby(['layer', groupby]).std().reset_index().rename(index=str, columns={'rel_amount': 'std_rel_amount'})

    opps = pd.concat([mopps, sopps], axis=1, sort=False).drop_duplicates()
    opps = opps.loc[:, ~opps.columns.duplicated()]
    return opps
