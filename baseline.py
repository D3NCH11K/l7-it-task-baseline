from __future__ import annotations
import argparse, hashlib, json, platform, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)

LABELS = ['FrontEnd', 'BackEnd', 'QA', 'DevOps', 'ML', 'UI/UX']
SEED = 42
EXPECTED_SHA256 = '90859af964559d1140fec298bf0b4f70e725dac3e7813e2062b2ea50fe9dfdf2'

def read_dataset(path: Path) -> tuple[pd.DataFrame, str]:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    lines = raw.decode('utf-8-sig').splitlines()
    rows=[]
    for line_no, line in enumerate(lines, 1):
        rec=line.strip()
        if rec.startswith('"') and rec.endswith('"'):
            rec=rec[1:-1]
        text, sep, label=rec.partition(';')
        if not sep:
            raise ValueError(f'Не разобрана строка {line_no}')
        rows.append((text,label))
    if not rows or rows[0] != ('text','label'):
        raise ValueError(f'Неверный заголовок CSV: {rows[:1]}')
    return pd.DataFrame(rows[1:], columns=['text','label']), sha

def run_baseline(data_path: str|Path, output_dir: str|Path, *, verify_hash: bool=True) -> dict:
    data_path, out=Path(data_path),Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    raw, source_sha=read_dataset(data_path)
    if verify_hash and source_sha != EXPECTED_SHA256:
        raise ValueError(f'Контрольная сумма источника изменилась: {source_sha}')
    clean=raw.copy()
    clean['text']=clean.text.astype(str).str.strip()
    clean['label']=clean.label.astype(str).str.strip()
    clean['norm_text']=clean.text.str.casefold().str.replace(r'\s+',' ',regex=True)
    assert not clean.text.eq('').any() and not clean.label.eq('').any()
    assert clean.label.isin(LABELS).all()
    conflicts=int((clean.groupby('norm_text').label.nunique()>1).sum())
    assert conflicts == 0, f'Конфликты label на одинаковом нормализованном тексте: {conflicts}'
    unique=clean.drop_duplicates('norm_text',keep='first').copy()
    unique['record_id']=unique.norm_text.map(lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:16])
    train, holdout=train_test_split(unique,test_size=.30,random_state=SEED,stratify=unique.label)
    validation, test=train_test_split(holdout,test_size=.50,random_state=SEED,stratify=holdout.label)
    assert not (set(train.record_id)&set(validation.record_id) or set(train.record_id)&set(test.record_id) or set(validation.record_id)&set(test.record_id))
    train_counts=train.label.value_counts().reindex(LABELS,fill_value=0)
    majority=str(train_counts.idxmax())
    y_true=validation.label.to_numpy(); y_pred=[majority]*len(validation)
    p,r,f,s=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,zero_division=0)
    metrics={
      'dataset_version':'tasks_v1.0','source_sha256':source_sha,
      'source_rows':int(len(raw)),'deduplicated_rows':int(len(unique)),
      'exact_duplicates_removed':int(len(raw)-len(unique)),'conflicting_text_groups':conflicts,
      'train_n':int(len(train)),'validation_n':int(len(validation)),'test_n':int(len(test)),
      'majority_class':majority,'majority_train_count':int(train.label.value_counts()[majority]),
      'accuracy':float(accuracy_score(y_true,y_pred)),
      'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),
      'macro_f1':float(f1_score(y_true,y_pred,labels=LABELS,average='macro',zero_division=0)),
      'weighted_f1':float(f1_score(y_true,y_pred,labels=LABELS,average='weighted',zero_division=0)),
      'randomness':'baseline deterministic; seed 42 applies only to reproducible stratified split',
    }
    pd.DataFrame({'label':LABELS,'precision':p,'recall':r,'f1':f,'support':s}).to_csv(out/'per_class_metrics.csv',index=False)
    pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(confusion_matrix(y_true,y_pred,labels=LABELS),index=LABELS,columns=LABELS).to_csv(out/'confusion_matrix.csv',index_label='true_label')
    cases=validation[['record_id','text','label']].copy().rename(columns={'label':'true_label'})
    cases['predicted_label']=majority; cases['correct']=cases.true_label.eq(majority)
    cases['split']='validation'
    cases.to_csv(out/'validation_predictions.csv',index=False,encoding='utf-8-sig')
    good=cases[cases.correct].sort_values('record_id').head(10)
    bad=(cases[~cases.correct].sort_values(['true_label','record_id'])
         .groupby('true_label',group_keys=False).head(2).sort_values(['true_label','record_id']).head(10))
    good.to_csv(out/'successful_cases.csv',index=False,encoding='utf-8-sig')
    bad.to_csv(out/'error_cases.csv',index=False,encoding='utf-8-sig')
    settings={
      'baseline':'majority class (most frequent label in train)', 'dataset_version':'tasks_v1.0',
      'source_sha256':source_sha, 'split':'stratified 70/15/15 after normalized exact-text deduplication',
      'seed':SEED, 'train_rows':len(train),'validation_rows':len(validation),'test_rows':len(test),
      'target':'label','text_features_used':False,'test_touched':False,
      'tie_break':'first class in LABELS order among classes tied for maximum train frequency',
      'tokenizer':'none','vectorizer':'none','hyperparameters':{},'randomness':'deterministic prediction; seed applies to split only'
    }
    (out/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    environment={'python':platform.python_version(),'pandas':pd.__version__}
    import numpy, sklearn
    environment.update({'numpy':numpy.__version__,'scikit_learn':sklearn.__version__})
    (out/'environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.concat([train.assign(split='train'),validation.assign(split='validation'),test.assign(split='test')])\
      [['record_id','text','label','split']].to_csv(out/'recreated_split.csv',index=False,encoding='utf-8-sig')
    result={'metrics':metrics,'per_class':pd.DataFrame({'label':LABELS,'precision':p,'recall':r,'f1':f,'support':s}).to_dict('records'),'environment':environment,'output':str(out)}
    (out/'run_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Reproducible majority-class baseline for tasks_dataset')
    parser.add_argument('--data',default='tasks_dataset.csv')
    parser.add_argument('--output',default='results')
    parser.add_argument('--skip-hash-check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run_baseline(args.data,args.output,verify_hash=not args.skip_hash_check),ensure_ascii=False,indent=2))
