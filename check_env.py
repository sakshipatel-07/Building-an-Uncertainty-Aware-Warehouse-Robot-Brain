import sys

packages = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'skfuzzy', 'pgmpy', 'torch']
for p in packages:
    try:
        mod = __import__(p)
        print(f"{p}: OK ({getattr(mod, '__version__', 'unknown')})")
    except Exception as e:
        print(f"{p}: NOT FOUND ({e})")
