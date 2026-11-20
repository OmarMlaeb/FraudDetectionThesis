import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis_common import *

def main():
    selected=read('selected_runs.json');tables=json.loads((ROOT/'audit/thesis/final_tables.json').read_text(encoding='utf-8'))['tables'];alignment=[]
    for number in [5,6,7,8]:
        table=next(t for t in tables if t['number']==f'4.{number}')
        for row in table['rows']:
            dataset='Elliptic' if number==8 else 'IEEE-CIS'
            if number in [5,6]:
                construction={'Main':'main','Card':'card','Address':'address','Email domain':'email_domain','Identifier, 24 h':'identifier_time_window-24h'}[row[1]]
                model=row[0]+'-'+construction
            else:model=row[0]+('-main' if number==7 and row[0] in ['GCN','GraphSAGE','GAT'] else '')
            rs=[r for r in selected if r['dataset']==dataset and r['model']==model]
            assert sorted(r['seed'] for r in rs)==SEEDS
            fields=['pr_auc','f1_score','recall'] if number==5 else ['false_positive'] if number==6 else ['recall','precision','f1_score','pr_auc','roc_auc']
            expected=row[2:] if number in [5,6] else row[1:]
            observed=[]
            for metric in fields:
                x=np.array([float(r[metric]) for r in rs])
                observed.append(f'{x.mean():.4f}±{x.std(ddof=1):.4f}' if number==5 else f'{x.mean():,.1f}±{x.std(ddof=1):,.1f}' if number==6 else f'{x.mean():.4f}')
            assert observed==expected,(number,model,observed,expected)
            alignment.append({'table':f'4.{number}','model':model,'dataset':dataset,'seeds':SEEDS,'matches_all_printed_values':True,'expected':expected,'recomputed':observed,'source_lines':[{'source':r['source'],'line':r['source_line'],'seed':r['seed']} for r in rs]})
    dump('thesis_tables_4_5_to_4_8_alignment.json',alignment)
    rows=read('subgroup_per_seed.json')
    for r in rows:
        assert r['positive']+r['negative']==r['n']
        assert r['true_positive']+r['false_negative']==r['positive']
        assert r['true_negative']+r['false_positive']==r['negative']
        if r['n']:assert r['fraud_rate']==r['positive']/r['n']
        else:assert r['pr_auc'] is None and r['f1_score'] is None and r['recall'] is None
    stats=read('wilcoxon_holm_inference_verified.json')
    for family in {r['family_id'] for r in stats}:
        rs=sorted([r for r in stats if r['family_id']==family],key=lambda r:r['p_raw']);n=len(rs)
        previous=0.
        for i,r in enumerate(rs):
            expected=max(previous,min(1.,(n-i)*r['p_raw']))
            assert r['p_holm']==expected and r['p_holm']>=r['p_raw']
            previous=expected
    rep=list(csv.DictReader((OUT/'representative_seed51.csv').open()))
    assert len(rep)==12 and all(r['seed']=='51' and r['checkpoint_verified']=='True' for r in rep)
    assert all(r['model'].endswith('-main') for r in rep if r['dataset']=='IEEE-CIS' and r['architecture'] in ['GCN','GraphSAGE','GAT'])
    dump('output_validation.json',{'thesis_table_rows_matched':len(alignment),'subgroup_rows_checked':len(rows),'holm_comparisons_checked':len(stats),'representative_rows_checked':len(rep),'passed':True})
    print('Validated thesis aggregate alignment, group counts/undefined metrics, Holm calculations, and representative seeds.',flush=True)

if __name__=='__main__':main()
