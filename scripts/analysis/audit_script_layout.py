"""Validate CLI/source paths and that worker relocation preserves computation."""
import _bootstrap
import ast
import hashlib
import json
from pathlib import Path

ROOT = _bootstrap.ROOT


def canonical_tree(source):
    """Undo only the recorded import/root/path plumbing before comparing ASTs."""
    manifest=json.loads((ROOT/'scripts/layout.json').read_text())
    reverse={r['new_path']:r['old_path'] for r in manifest['scripts']}
    class Normalize(ast.NodeTransformer):
        def visit_Import(self,node):
            if len(node.names)==1 and node.names[0].name=='_bootstrap':return None
            return node
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='_bootstrap' and node.attr=='ROOT':
                return ast.parse('Path(__file__).resolve().parent',mode='eval').body
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in reverse:node.value=reverse[node.value]
            return node
    return ast.dump(Normalize().visit(ast.parse(source)),include_attributes=False)


def main():
    manifest=json.loads((ROOT/'scripts/layout.json').read_text());checks=[]
    def check(name,ok):
        checks.append({'check':name,'passed':bool(ok)})
    check('No Python scripts left in repository root',not list(ROOT.glob('*.py')))
    for row in manifest['scripts']:
        check(row['new_path'],(ROOT/row['new_path']).is_file() and not (ROOT/row['old_path']).exists())
    for old,new in manifest['shell_scripts'].items():check(new,(ROOT/new).is_file() and not (ROOT/old).exists())
    modules={p.stem:p for p in (ROOT/'scripts').rglob('*.py') if p.name!='_bootstrap.py'}
    modules['_bootstrap']=ROOT/'scripts/_bootstrap.py'
    for path in (ROOT/'scripts').rglob('*.py'):
        tree=ast.parse(path.read_text());check(str(path.relative_to(ROOT))+' parses',True)
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith(('scripts/analysis/','scripts/experiments/','scripts/reports/','scripts/setup/')) and node.value.endswith(('.py','.sh')):
                check(node.value+' exists',(ROOT/node.value).is_file())
            if isinstance(node,ast.ImportFrom) and node.module in modules:check(node.module+' local import',modules[node.module].is_file())
    workers=[]
    for name in ('parallel_sd_worker.py','sweep_worker.py'):
        row=next(r for r in manifest['scripts'] if r['old_path']==name)
        archive=ROOT/'logs/script_layout_20261107'/(row['original_sha256']+'.py')
        current=ROOT/row['new_path'];original=archive.read_bytes()
        identical=hashlib.sha256(original).hexdigest()==row['original_sha256'] and canonical_tree(current.read_text())==ast.dump(ast.parse(original),include_attributes=False)
        check(name+' unchanged computation AST',identical)
        workers.append({'worker':row['new_path'],'archived_original':str(archive.relative_to(ROOT)),'archived_SHA256':row['original_sha256'],'current_SHA256':hashlib.sha256(current.read_bytes()).hexdigest(),'same_computation_AST':identical})
    output={'passed':all(c['passed'] for c in checks),'checks':checks,'workers':workers}
    (ROOT/'logs/script_layout_20261107/validation.json').write_text(json.dumps(output,indent=2))
    print(json.dumps({'passed':output['passed'],'checks':len(checks),'workers':workers,'failures':[c for c in checks if not c['passed']]},indent=2))
    raise SystemExit(0 if output['passed'] else 1)


if __name__=='__main__':main()
