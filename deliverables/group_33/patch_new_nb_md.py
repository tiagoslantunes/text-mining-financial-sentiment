"""Patch dead src/*.py / scripts/*.py references in new_nb.ipynb markdown cells."""
import json, re, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Reuse the patch function from merge_notebooks — import only that function
# by exec-ing just its definition block after stripping the rest
with open('merge_notebooks.py', encoding='utf-8') as f:
    mscript = f.read()

# Extract only the patch_md_src_refs function
fn_start = mscript.index('def patch_md_src_refs')
fn_end   = mscript.index('\n# ── Build merged', fn_start)
fn_code  = mscript[fn_start:fn_end]
globs = {'re': re}
exec(fn_code, globs)
patch_md_src_refs = globs['patch_md_src_refs']

with open('new_nb.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'markdown':
        continue
    src = ''.join(cell.get('source', []))
    new_src = patch_md_src_refs(src)
    if new_src != src:
        cell['source'] = [new_src]
        print(f'  [{i:03d}] Patched markdown cell')
        patched += 1

print(f'\n{patched} markdown cells patched in new_nb.ipynb')

# Verify no dead refs remain
dead_patterns = [r'src/[\w.]+\.py', r'scripts/[\w.]+\.py']
remaining = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'markdown': continue
    src = ''.join(cell.get('source', []))
    hits = [m.group() for p in dead_patterns for m in re.finditer(p, src)]
    if hits:
        remaining.append((i, hits))

if remaining:
    print('\nREMAINING DEAD REFS:')
    for i, hits in remaining:
        print(f'  cell {i:03d}: {hits}')
else:
    print('Verification: no dead refs remaining in new_nb.ipynb')

with open('new_nb.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Saved: new_nb.ipynb')
