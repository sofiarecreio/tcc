"""Lista os downloads presentes e os que faltam, sem ler tabelas grandes."""
from pathlib import Path
from caminhos_dados import localizar_tabela

TABELAS = {
    'mimic-iv': ['patients', 'admissions', 'diagnoses_icd', 'icustays', 'chartevents', 'labevents',
                 'inputevents', 'outputevents', 'd_items'],
    'mimic-iv-structured-measurement': ['structured-measurement'],
    'eicu': ['patient', 'diagnosis', 'vitalPeriodic', 'vitalAperiodic', 'lab'],
}

if __name__ == '__main__':
    for base, tabelas in TABELAS.items():
        print(f'\n{base}:')
        for nome in tabelas:
            try:
                print(f'  OK: {localizar_tabela(Path("data") / base, nome)}')
            except (FileNotFoundError, ValueError) as e:
                print(f'  PENDENTE: {e}')
