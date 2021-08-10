#!/usr/bin/env python
# coding: utf-8

# #  common code

# In[1]:


import os
import matplotlib
from scipy import stats
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame, Series
from polybenchc_common import MEASUREMENTS_DIR, ANALYSIS_DIR, get_all_measurements, aggregate_measurements,     normalize_measurements_by_names, BENCHMARK_TYPE_BINARY_SIZE, BENCHMARK_TYPE_INITIALIZATION_TIME,     BENCHMARK_TYPE_EXECUTION_TIME, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_WAVM, RUNTIME_WAVM_SIMD,     RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED, RUNTIME_WASMER, COMPILE_TARGET_WASM,     COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_JS, RUNTIME_X86_64_GENERIC, COMPILE_TARGET_EMCC_WASM,     RUNTIME_WASMER_LLVM, aggregate_data, BENCHMARK_TYPE_GZIP_BINARY_SIZE, RUNTIME_NODEJS_WASM,     RUNTIME_SPIDERMONKEY_WASM, RUNTIME_X86_64_NATIVE, mean_measurements, gmean_meaned_measurements,     COMPILE_TARGET_WASM_SIMD, normalize_by_index
from common import get_confidence_error
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import seaborn as sns

RUNTIME_NAMES = {
    RUNTIME_X86_64_GENERIC: 'x86-64',
    RUNTIME_X86_64_NATIVE: 'x86-64 (native)',
    RUNTIME_SPIDERMONKEY: 'SpiderMonkey (asm.js)',
    RUNTIME_SPIDERMONKEY_WASM: 'SpiderMonkey (WASM)',
    RUNTIME_NODEJS: 'Node.js (asm.js)',
    RUNTIME_NODEJS_WASM: 'Node.js (WASM)',
    RUNTIME_WASMER:'Wasmer',
    RUNTIME_WASMTIME_CRANELIFT:'Wasmtime',
    RUNTIME_WAVM:'WAVM',
    RUNTIME_WAVM_SIMD:'WAVM (SIMD)'
}
RUNTIME_COLORS = {
    RUNTIME_X86_64_GENERIC: '#455a64',
    RUNTIME_X86_64_NATIVE: '#000000',
    RUNTIME_NODEJS:'#65ba69',
    RUNTIME_NODEJS_WASM: '#5c6bc0',
    RUNTIME_SPIDERMONKEY: '#ef5350',
    RUNTIME_SPIDERMONKEY_WASM: '#ffa726',
    RUNTIME_WAVM: '#ffc107', 
    RUNTIME_WAVM_SIMD: '#000000',
    RUNTIME_WASMTIME_CRANELIFT:'#26a69a',
    RUNTIME_WASMER:'#7385eb', 
}
COMPILE_TARGET_NAMES = {
    COMPILE_TARGET_X86_64_GENERIC: 'x86-64 from clang',
    COMPILE_TARGET_WASM: 'WASM from clang',
    COMPILE_TARGET_EMCC_WASM: 'WASM from emcc',
    COMPILE_TARGET_JS: 'asm.js from emcc',
    COMPILE_TARGET_WASM_SIMD: 'WASM (SIMD) from clang',
}
COMPILE_TARGET_COLORS = {
    COMPILE_TARGET_X86_64_GENERIC: '#455a64',
    COMPILE_TARGET_WASM: '#26a69a',
    COMPILE_TARGET_EMCC_WASM: '#5c6bc0',
    COMPILE_TARGET_JS: '#ffc107',
}


# # all measurements

# In[2]:


all = get_all_measurements()
all


# # geometric mean of execution time per runtime

# In[3]:


def get_virtual_combined_runtime_data(df: DataFrame, runtimes, runtime_name) -> DataFrame:
    df = df[df['name'].isin(runtimes)]
    df['name'] = runtime_name
    return df

def get_virtual_asmjs_runtime_data(df: DataFrame) -> DataFrame:
    return get_virtual_combined_runtime_data(df, [RUNTIME_SPIDERMONKEY, RUNTIME_NODEJS], 'virtualasmjs')

runtimes = ['virtualasmjs', 'vemccwasm', RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_WASMER, RUNTIME_WASMTIME_CRANELIFT]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = df.append(get_virtual_asmjs_runtime_data(df))
df = df.append(get_virtual_combined_runtime_data(df, [RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM], 'vemccwasm'))
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.drop(columns=['type'])
df = df.set_index('name')
df.index.name = 'runtime'
df = df.filter(runtimes, axis=0)
df = df.sort_values(by=['gmean'])
df['x86 normal'] = normalize_by_index(df, RUNTIME_X86_64_GENERIC, 'gmean')
df['asm.js normal'] = normalize_by_index(df, 'virtualasmjs', 'gmean')
df['asm.js node normal'] = normalize_by_index(df, RUNTIME_NODEJS, 'gmean')
df['asm.js spidermonkey normal'] = normalize_by_index(df, RUNTIME_SPIDERMONKEY, 'gmean')
df = df.rename(index=RUNTIME_NAMES)
df


# In[37]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_WASMER, RUNTIME_WASMTIME_CRANELIFT]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df = df.filter(runtimes, axis=0)
df = df.sort_values(by=['gmean'])
colors = df.index.to_series().map(RUNTIME_COLORS)
df = df.rename(index=RUNTIME_NAMES)
means = df['gmean']
chart = means.plot.barh(width=0.9, alpha=1, align='center', color=colors)
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0)
plt.xticks([0,1,2,3,4])
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend().remove()
fig = chart.get_figure()
fig.set_size_inches(5,3)
name = 'execution-time-gmean'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # wavm vs x86

# In[37]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_WAVM]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = df[df['name'].isin(runtimes)]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = df[df['name'] == RUNTIME_WAVM]
df = mean_measurements(df)
df


# In[6]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY_WASM]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = df[df['name'].isin(runtimes)]
#df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = mean_measurements(df)
df = df.groupby(['test']).apply(lambda r: pd.Series(
    {
     'wavm div': r[r['name'] == RUNTIME_WAVM].iloc[0]['mean'] - r[r['name'] == RUNTIME_X86_64_GENERIC].iloc[0]['mean'] + r[r['name'] == RUNTIME_X86_64_GENERIC].iloc[0]['95error'] + r[r['name'] == RUNTIME_WAVM].iloc[0]['95error'],
     'node div': r[r['name'] == RUNTIME_NODEJS_WASM].iloc[0]['mean'] - r[r['name'] == RUNTIME_X86_64_GENERIC].iloc[0]['mean'] + r[r['name'] == RUNTIME_X86_64_GENERIC].iloc[0]['95error'] + r[r['name'] == RUNTIME_NODEJS_WASM].iloc[0]['95error'],
     'spidermonkey div': r[r['name'] == RUNTIME_SPIDERMONKEY_WASM].iloc[0]['mean'] - r[r['name'] == RUNTIME_X86_64_GENERIC].iloc[0]['mean'] + r[r['name'] == RUNTIME_X86_64_GENERIC].iloc[0]['95error'] + r[r['name'] == RUNTIME_SPIDERMONKEY_WASM].iloc[0]['95error'],
     }))
df


# # execution time

# In[5]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WASMER, RUNTIME_WAVM]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES).reset_index().rename(columns={'test': 'Algorithm'})
df.columns.name = None
df.index.name = None
name = 'execution-time'
df.to_csv(os.path.join(ANALYSIS_DIR, name + '.csv'), index=False)
df


# # execution time normalized to x86-64

# In[30]:


runtimes = [RUNTIME_WAVM, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WASMER]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES)
errors = df.pivot(index='test', columns='name', values='95error').filter(runtimes).rename(columns=RUNTIME_NAMES)
chart = means.plot.barh(width=0.9, xerr=errors, alpha=1, align='center', color=pd.Series(runtimes).map(RUNTIME_COLORS), error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.set_xscale('symlog', linthreshx=6.4)
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.legend(title='')
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(handles=[Line2D([0], [0], color='black', alpha=0.7, linewidth=1, linestyle='--')] + handles, labels=[RUNTIME_NAMES[RUNTIME_X86_64_GENERIC]] + lables, frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
plt.axvline(x=1, color='black', alpha=0.7, linewidth=1, linestyle='--')
plt.xticks([0,1,2,4,5,10,15])
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
name = 'execution-time'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # execution time WAVM vs. WAVM (SIMD) normalized to x86-64

# In[7]:


runtimes = [RUNTIME_WAVM, RUNTIME_WAVM_SIMD]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES)
errors = df.pivot(index='test', columns='name', values='95error').filter(runtimes).rename(columns=RUNTIME_NAMES)
chart = means.plot.barh(width=0.9, xerr=errors, alpha=1, align='center', color=pd.Series(runtimes).map(RUNTIME_COLORS), error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.set_xscale('symlog', linthreshx=6.4)
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.legend(title='')
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(handles=[Line2D([0], [0], color='black', alpha=0.7, linewidth=1, linestyle='--')] + handles, labels=[RUNTIME_NAMES[RUNTIME_X86_64_GENERIC]] + lables, frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
plt.axvline(x=1, color='black', alpha=0.7, linewidth=1, linestyle='--')
plt.xticks([0,1,2,3])
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
name = 'execution-time-simd'
#fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
#fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # execution time of 4 fastest runtimes normalized to x86-64

# In[32]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY_WASM]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES).reset_index().rename(columns={'test': 'Algorithm'})
df.columns.name = None
df.index.name = None
name = 'execution-time-top-4'
#df.to_csv(os.path.join(ANALYSIS_DIR, name + '.csv'), index=False, float_format='%.5f')
df


# In[4]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY_WASM]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES)
errors = df.pivot(index='test', columns='name', values='95error').filter(runtimes).rename(columns=RUNTIME_NAMES)
chart = means.plot.barh(width=0.85, xerr=errors, alpha=1, align='center', color=pd.Series(runtimes).map(RUNTIME_COLORS), error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
#chart.set_xscale('symlog', linthreshx=6.4)
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.legend(title='')
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
plt.xticks([0,1,2,3])
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
name = 'execution-time-top-4'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # execution time normalized to asm.js 

# In[10]:


runtimes = [RUNTIME_WAVM, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WASMER]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY])
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES)
errors = df.pivot(index='test', columns='name', values='95error').filter(runtimes).rename(columns=RUNTIME_NAMES)
chart = means.plot.barh(width=0.9, xerr=errors, alpha=1, align='center', color=pd.Series(runtimes).map(RUNTIME_COLORS), error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(handles=[Line2D([0], [0], color='black', alpha=0.7, linewidth=1, linestyle='--')] + handles, labels=['asm.js'] + lables, title='', loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
plt.axvline(x=1, color='black', alpha=0.7, linewidth=1, linestyle='--')
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
name = 'execution-time-js'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # asm.js vs wasm - node.js

# In[21]:


runtimes = [RUNTIME_NODEJS_WASM, RUNTIME_NODEJS]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_NODEJS])
df = aggregate_measurements(df)
df = df[df['name'].isin(runtimes)]
df = df.pivot(index='test', columns='name', values='mean')
print('gmean: %f' % stats.gmean(df[RUNTIME_NODEJS_WASM].values))
df


# # asm.js vs wasm - spidermonkey

# In[22]:


runtimes = [RUNTIME_SPIDERMONKEY_WASM, RUNTIME_SPIDERMONKEY]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_SPIDERMONKEY])
df = aggregate_measurements(df)
df = df[df['name'].isin(runtimes)]
df = df.pivot(index='test', columns='name', values='mean')
print('gmean: %f' % stats.gmean(df[RUNTIME_SPIDERMONKEY_WASM].values))
df


# # asm.js vs wasm

# In[5]:


runtimes = [RUNTIME_NODEJS_WASM, RUNTIME_NODEJS]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_NODEJS])
df = aggregate_measurements(df)
df = df[df['name'].isin(runtimes)]
df = df.pivot(index='test', columns='name', values='mean')
node = df[RUNTIME_NODEJS_WASM].values
runtimes = [RUNTIME_SPIDERMONKEY_WASM, RUNTIME_SPIDERMONKEY]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_SPIDERMONKEY])
df = aggregate_measurements(df)
df = df[df['name'].isin(runtimes)]
df = df.pivot(index='test', columns='name', values='mean')
spidermonkey = df[RUNTIME_SPIDERMONKEY_WASM].values
ratios = []
ratios.extend(node)
ratios.extend(spidermonkey)
print(ratios)
gmean = stats.gmean(ratios)
print(gmean)


# # execution time asm.js vs. wasm on node.js and spidermonkey normalized to x86-64

# In[9]:


runtimes = [RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY_WASM]
df = all[all['type'] == BENCHMARK_TYPE_EXECUTION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(runtimes).rename(columns=RUNTIME_NAMES)
errors = df.pivot(index='test', columns='name', values='95error').filter(runtimes).rename(columns=RUNTIME_NAMES)
chart = means.plot.barh(width=0.85, xerr=errors, alpha=1, align='center', color=pd.Series(runtimes).map(RUNTIME_COLORS), error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.set_xscale('symlog', linthreshx=5.5)
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.legend(title='')
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(handles=[Line2D([0], [0], color='black', alpha=0.7, linewidth=1, linestyle='--')] + handles, labels=['x86-64'] + lables, frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
plt.axvline(x=1, color='black', alpha=0.7, linewidth=1, linestyle='--')
#plt.xticks([0,0.25,0.5,0.75,1,1.25,1.5])
plt.xticks([0,1,2,5,10])
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
name = 'execution-time-node-spidermonkey'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # initialization time

# In[12]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WASMER, RUNTIME_WAVM]
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean')
df = df.filter(runtimes).rename(columns=RUNTIME_NAMES)
df.columns.name = None
df.index.name = 'Algorithm'
df.to_csv(os.path.join(ANALYSIS_DIR, 'initialization-time.csv'))
df


# # initialization time

# In[39]:


columns = {
    RUNTIME_WAVM:'WAVM',
    #RUNTIME_WAVM_SIMD:'WAVM (SIMD enabled)',
    RUNTIME_WASMTIME_CRANELIFT:'Wasmtime',
    #RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED:'Wasmtime (optimized)',
    RUNTIME_WASMER:'Wasmer',
    #RUNTIME_WASMER_LLVM:'Wasmer (LLVM backend)'
}
colors = ['#ffc107', '#26a69a','#7385eb', '#4ad44a', 'red']
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(columns.keys()).rename(columns=columns)
errors = df.pivot(index='test', columns='name', values='95error').filter(columns.keys()).rename(columns=columns)
chart = means.plot.barh(width=0.9, xerr=errors, alpha=1, align='center', color=colors, error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(title='', loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
fig.savefig(os.path.join(ANALYSIS_DIR, 'initialization-time.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, 'initialization-time.pgf'), bbox_inches='tight')
plt.show()


# # initialization time (in seconds)

# In[14]:


df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = aggregate_data(df, 'value', ['name'])
df = df.sort_values(by=['mean'])
df


# # geometric mean of execution time per runtime

# In[14]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_WASMER, RUNTIME_WASMTIME_CRANELIFT]
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df = df.drop(columns=['type'])
df = df.apply(lambda v: v * 1000) # s to ms
df = df.filter(runtimes, axis=0)
df = df.rename(index=RUNTIME_NAMES)
df = df.sort_values(by=['gmean'])
df


# # geometric mean of execution time per runtime normalized to x86-64

# In[3]:


def get_virtual_combined_runtime_data(df: DataFrame, runtimes, runtime_name) -> DataFrame:
    df = df[df['name'].isin(runtimes)]
    df['name'] = runtime_name
    return df

def get_virtual_asmjs_runtime_data(df: DataFrame) -> DataFrame:
    return get_virtual_combined_runtime_data(df, [RUNTIME_SPIDERMONKEY, RUNTIME_NODEJS], 'virtualasmjs')

runtimes = ['virtualasmjs', 'vemccwasm', RUNTIME_X86_64_GENERIC, RUNTIME_WAVM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM, RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY, RUNTIME_WASMER, RUNTIME_WASMTIME_CRANELIFT]
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = df.append(get_virtual_asmjs_runtime_data(df))
df = df.append(get_virtual_combined_runtime_data(df, [RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM], 'vemccwasm'))
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.drop(columns=['type'])
df = df.set_index('name')
df.index.name = 'runtime'
df = df.filter(runtimes, axis=0)
df = df.sort_values(by=['gmean'])
df['x86 normal'] = normalize_by_index(df, RUNTIME_X86_64_GENERIC, 'gmean')
#df['asm.js normal'] = normalize_by_index(df, 'virtualasmjs', 'gmean')
#df['asm.js node normal'] = normalize_by_index(df, RUNTIME_NODEJS, 'gmean')
#df['asm.js spidermonkey normal'] = normalize_by_index(df, RUNTIME_SPIDERMONKEY, 'gmean')
df['wasm node normal'] = normalize_by_index(df, RUNTIME_NODEJS_WASM, 'gmean')
df['wasm spidermonkey normal'] = normalize_by_index(df, RUNTIME_SPIDERMONKEY_WASM, 'gmean')
df = df.rename(index=RUNTIME_NAMES)
df


# In[6]:


runtimes = [RUNTIME_X86_64_GENERIC, RUNTIME_SPIDERMONKEY, RUNTIME_NODEJS, RUNTIME_WASMER, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WAVM, RUNTIME_SPIDERMONKEY_WASM, RUNTIME_NODEJS_WASM]
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df = df.filter(runtimes, axis=0)
df = df.sort_values(by=['gmean'])
colors = df.index.to_series().map(RUNTIME_COLORS)
df = df.rename(index=RUNTIME_NAMES)
means = df['gmean']
means = means.apply(lambda v: v * 1000)
chart = means.plot.barh(width=0.9, alpha=1, align='center', color=colors)
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0,400)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('time in ms')
plt.ylabel('')
#plt.xticks([0,0.05,0.1,0.2,0.3,0.4])
plt.legend().remove()
fig = chart.get_figure()
fig.set_size_inches(5,3)
name = 'initialization-time-gmean'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # initialization time relative to native (in percent)

# In[16]:


df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = aggregate_data(df, 'value', ['name'])
df


# # initialization time relative to asm.js (in percent)

# In[18]:


df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY])
df = aggregate_data(df, 'value', ['name'])
df


# In[19]:


columns = {
    RUNTIME_WAVM:'WAVM',
    #RUNTIME_WAVM_SIMD:'WAVM (SIMD enabled)',
    RUNTIME_WASMTIME_CRANELIFT:'Wasmtime',
    #RUNTIME_WASMTIME_CRANELIFT_OPTIMIZED:'Wasmtime (optimized)',
    RUNTIME_WASMER:'Wasmer'
}
colors = ['#ffc107', '#26a69a','#7385eb', '#4ad44a', 'red']
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = normalize_measurements_by_names(df, [RUNTIME_NODEJS, RUNTIME_SPIDERMONKEY])
df = aggregate_measurements(df)
means = df.pivot(index='test', columns='name', values='mean').filter(columns.keys()).rename(columns=columns)
errors = df.pivot(index='test', columns='name', values='95error').filter(columns.keys()).rename(columns=columns)
chart = means.plot.barh(width=0.9, xerr=errors, alpha=1, align='center', color=colors, error_kw={'ecolor':'black', 'elinewidth':0.7, 'capsize':1.5})
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_major_formatter(PercentFormatter(1.0))
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('')
plt.ylabel('')
plt.legend(handles=[Line2D([0], [0], color='black', alpha=0.7, linewidth=1, linestyle='--')] + handles, labels=['asm.js'] + lables, title='', loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
plt.axvline(x=1, color='black', alpha=0.7, linewidth=1, linestyle='--')
fig = chart.get_figure()
fig.set_size_inches(6.2,10)
fig.savefig(os.path.join(ANALYSIS_DIR, 'initialization-time-js.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, 'initialization-time-js.pgf'), bbox_inches='tight')
plt.show()


# In[17]:


targets=[RUNTIME_X86_64_GENERIC, RUNTIME_NODEJS, RUNTIME_NODEJS_WASM, RUNTIME_SPIDERMONKEY, RUNTIME_SPIDERMONKEY_WASM]
colors = pd.Series(targets).map(RUNTIME_COLORS)
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = aggregate_measurements(df)
mean = df.pivot(index='test', columns='name', values='mean').filter(targets).rename(columns=RUNTIME_NAMES).apply(lambda v: v * 1000)
errors = df.pivot(index='test', columns='name', values='95error').filter(targets).rename(columns=RUNTIME_NAMES).apply(lambda v: v * 1000)
fig, ax = plt.subplots()
for i in range(len(targets)):
    #plt.scatter(mean.iloc[:,i], mean.index, label=mean.columns[i], color=colors[i], alpha=0.8)
    plt.errorbar(mean.iloc[:,i], mean.index, xerr=errors.iloc[:,i], label=mean.columns[i], fmt='|', color=colors[i], alpha=0.8)
plt.xlim(0)
ax.invert_yaxis()
ax.tick_params(direction='in')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('none')
ax.xaxis.grid(True)
ax.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
ax.margins(y=0.02)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('time in ms')
plt.ylabel('')
plt.legend(title='', loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.12), frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig.set_size_inches(6.2,9)
name = 'initialization-time-node-spidermonkey'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# In[27]:


targets=[RUNTIME_X86_64_GENERIC, RUNTIME_WASMER, RUNTIME_WASMTIME_CRANELIFT, RUNTIME_WAVM]
colors = pd.Series(targets).map(RUNTIME_COLORS)
df = all[all['type'] == BENCHMARK_TYPE_INITIALIZATION_TIME]
df = aggregate_measurements(df)
mean = df.pivot(index='test', columns='name', values='mean').filter(targets).rename(columns=RUNTIME_NAMES).apply(lambda v: v * 1000)
errors = df.pivot(index='test', columns='name', values='95error').filter(targets).rename(columns=RUNTIME_NAMES).apply(lambda v: v * 1000)
fig, ax = plt.subplots()
for i in range(len(targets)):
    #plt.scatter(mean.iloc[:,i], mean.index, label=mean.columns[i], color=colors[i], alpha=0.8)
    plt.errorbar(mean.iloc[:,i], mean.index, xerr=errors.iloc[:,i], label=mean.columns[i], fmt='|', color=colors[i], alpha=1)
plt.xlim(0)
ax.invert_yaxis()
ax.tick_params(direction='in')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('none')
ax.xaxis.grid(True)
ax.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
ax.margins(y=0.02)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('time in ms')
plt.ylabel('')
plt.legend(title='', loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.086), frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig.set_size_inches(6.2,9)
name = 'initialization-time-standalone'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # binary size measurements

# In[20]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean')
df.columns.name = None
df.index.name = 'Algorithm'
df = df.filter(targets)
df = df.rename(columns=COMPILE_TARGET_NAMES)
df.to_csv(os.path.join(ANALYSIS_DIR, 'binary-size.csv'), float_format='%.0f')
df


# In[21]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean')
df = df.filter(targets)
df = df.apply(lambda v: v / 1000)
df = df.rename(columns=COMPILE_TARGET_NAMES)
df.reset_index()


# In[10]:


df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df.index.name = 'target'
df = df.drop(columns=['type'])
df = df.apply(lambda v: v / 1000)
df['x86 normal'] = normalize_by_index(df, COMPILE_TARGET_X86_64_GENERIC, 'gmean')
df['asm.js normal'] = normalize_by_index(df, COMPILE_TARGET_JS, 'gmean')
df = df.rename(index=COMPILE_TARGET_NAMES)
df


# In[23]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df = df.filter(targets, axis=0)
df = df.sort_values(by=['gmean'])
colors = df.index.to_series().map(COMPILE_TARGET_COLORS)
df = df.rename(index=COMPILE_TARGET_NAMES)
df = df['gmean']
df = df.apply(lambda v: v / 1000)
chart = df.plot.barh(width=0.9, alpha=1, align='center', color=colors)
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0,70)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('size in KB')
plt.ylabel('')
#plt.xticks([0,80])
plt.legend().remove()
fig = chart.get_figure()
fig.set_size_inches(5,2)
name = 'binary-size-gmean'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# In[32]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
colors = pd.Series(targets).map(COMPILE_TARGET_COLORS)
df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean')
df = df.filter(targets)
df = df.apply(lambda v: v / 1000)
df = df.rename(columns=COMPILE_TARGET_NAMES)
fig, ax = plt.subplots()
for i in range(len(targets)):
    plt.scatter(df.iloc[:,i], df.index, label=df.columns[i], color=colors[i])
plt.xlim(0,80)
ax.invert_yaxis()
ax.tick_params(direction='in')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('none')
ax.xaxis.grid(True)
ax.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
ax.margins(y=0.02)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('size in KB')
plt.ylabel('')
plt.legend(title='', loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.086), frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig.set_size_inches(6.2,9)
name = 'binary-size'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # binary size measurements gzip
# 

# In[25]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean')
df.columns.name = None
df.index.name = 'Algorithm'
df = df.filter(targets)
df = df.rename(columns=COMPILE_TARGET_NAMES)
df.to_csv(os.path.join(ANALYSIS_DIR, 'binary-size-gzip.csv'), float_format='%.0f')
df


# # binary size (in bytes)

# In[26]:


df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = aggregate_data(df, 'value', ['name'])
df


# In[27]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_JS]
df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = df[df['name'].isin(targets)]
df['value'] = df['value'].apply(lambda v: v / 1000)
#native_mean = df.loc[df['name'] == COMPILE_TARGET_X86_64_GENERIC, 'value'].mean()
df['name'] = df['name'].apply(lambda n: COMPILE_TARGET_NAMES[n])
chart = sns.stripplot(x='value',y='name',data=df, hue='name', jitter=True, order=pd.Series(targets).map(COMPILE_TARGET_NAMES), alpha=0.3)
plt.xlabel('size in KB')
plt.ylabel('')
plt.yticks([])
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.legend(title='', loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig = chart.get_figure()
fig.set_size_inches(6.2, 3)
name = 'binary-size-dots'
#plt.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
#plt.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# # binary size compared to native (in percent)

# In[28]:


df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = normalize_measurements_by_names(df, [RUNTIME_X86_64_GENERIC])
df = aggregate_data(df, 'value', ['name'])
df


# # binary size compared to asm.js (in percent)

# In[29]:


df = all[all['type'] == BENCHMARK_TYPE_BINARY_SIZE]
df = normalize_measurements_by_names(df, [COMPILE_TARGET_JS])
df = aggregate_data(df, 'value', ['name'])
df


# # binary size gzip

# In[30]:


df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = aggregate_data(df, 'value', ['name'])
df


# ## normalized by x86-64

# In[31]:


df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = normalize_measurements_by_names(df, [COMPILE_TARGET_X86_64_GENERIC])
df = aggregate_data(df, 'value', ['name'])
df


# ## normalized by asm.js

# In[32]:


df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = normalize_measurements_by_names(df, [COMPILE_TARGET_JS])
df = aggregate_data(df, 'value', ['name'])
df


# ## compression ratio

# In[31]:


df = all[all['type'].isin([BENCHMARK_TYPE_GZIP_BINARY_SIZE, BENCHMARK_TYPE_BINARY_SIZE])]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index(['name','type'])
df = df.apply(lambda v: v / 1000)
df = df.unstack()
df.columns = df.columns.droplevel()
df['ratio'] = df.apply(lambda r: r[BENCHMARK_TYPE_BINARY_SIZE] / r[BENCHMARK_TYPE_GZIP_BINARY_SIZE], axis=1)
df = df.rename(index=COMPILE_TARGET_NAMES)
df


# In[11]:


df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df.index.name = 'target'
df = df.drop(columns=['type'])
df = df.apply(lambda v: v / 1000)
df['x86 normal'] = normalize_by_index(df, COMPILE_TARGET_X86_64_GENERIC, 'gmean')
df['asm.js normal'] = normalize_by_index(df, COMPILE_TARGET_JS, 'gmean')
df = df.rename(index=COMPILE_TARGET_NAMES)
df


# # geometric mean of gzip binary size

# In[34]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = mean_measurements(df)
df = gmean_meaned_measurements(df)
df = df.set_index('name')
df = df.filter(targets, axis=0)
df = df.sort_values(by=['gmean'])
colors = df.index.to_series().map(COMPILE_TARGET_COLORS)
df = df.rename(index=COMPILE_TARGET_NAMES)
df = df['gmean']
df = df.apply(lambda v: v / 1000)
chart = df.plot.barh(width=0.9, alpha=1, align='center', color=colors)
chart.invert_yaxis()
chart.tick_params(direction='in')
chart.xaxis.set_ticks_position('both')
chart.yaxis.set_ticks_position('none')
chart.xaxis.grid(True)
chart.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
handles, lables = chart.get_legend_handles_labels()
plt.xlim(0,25)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('size in KB')
plt.ylabel('')
#plt.xticks([0,80])
plt.legend().remove()
fig = chart.get_figure()
fig.set_size_inches(5,2)
name = 'binary-size-gzip-gmean'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# In[33]:


targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_EMCC_WASM, COMPILE_TARGET_JS]
colors = pd.Series(targets).map(COMPILE_TARGET_COLORS)
df = all[all['type'] == BENCHMARK_TYPE_GZIP_BINARY_SIZE]
df = aggregate_measurements(df)
df = df.pivot(index='test', columns='name', values='mean')
df = df.filter(targets)
df = df.apply(lambda v: v / 1000)
df = df.rename(columns=COMPILE_TARGET_NAMES)
fig, ax = plt.subplots()
for i in range(len(targets)):
    plt.scatter(df.iloc[:,i], df.index, label=df.columns[i], color=colors[i])
plt.xlim(0,25)
ax.invert_yaxis()
ax.tick_params(direction='in')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('none')
ax.xaxis.grid(True)
ax.grid(axis='x', linestyle='-', alpha=0.3, zorder=100)
ax.margins(y=0.02)
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.xlabel('size in KB')
plt.ylabel('')
plt.legend(title='', loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.086), frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig.set_size_inches(6.2,9)
name = 'binary-size-gzip'
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()


# In[36]:



targets=[COMPILE_TARGET_X86_64_GENERIC, COMPILE_TARGET_WASM, COMPILE_TARGET_JS]
df = all[all['type'].isin([BENCHMARK_TYPE_GZIP_BINARY_SIZE])]
df = df[df['name'].isin(targets)]
df['value'] = df['value'].apply(lambda v: v / 1000)
#native_mean = df.loc[df['name'] == COMPILE_TARGET_X86_64_GENERIC, 'value'].mean()
df['name'] = df['name'].apply(lambda n: COMPILE_TARGET_NAMES[n])
chart = sns.stripplot(x='value',y='name',data=df, hue='name', jitter=True, order=pd.Series(targets).map(COMPILE_TARGET_NAMES), alpha=0.3)
plt.xlabel('size in KB')
plt.ylabel('')
plt.yticks([])
plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False, 
})
plt.legend(title='', loc='upper right', frameon=True, fancybox=False, shadow=False, edgecolor='black').get_frame().set_linewidth(0.8)
fig = chart.get_figure()
fig.set_size_inches(6.2, 3)
name = 'binary-size-gzip-dots'
#plt.savefig(os.path.join(ANALYSIS_DIR, name + '.pdf'), bbox_inches='tight')
#plt.savefig(os.path.join(ANALYSIS_DIR, name + '.pgf'), bbox_inches='tight')
plt.show()

