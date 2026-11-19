"""Localização de tabelas CSV, preservando a estrutura dos downloads."""
from pathlib import Path


def localizar_tabela(raiz, nome):
    """Aceita CSV/GZIP na raiz ou subpastas; rejeita versões ambíguas."""
    raiz = Path(raiz)
    nome = Path(str(nome).replace("\\", "/")).name
    if nome.endswith(".gz"):
        nome = nome[:-3]
    if not nome.endswith(".csv"):
        nome += ".csv"
    encontrados = sorted(set(raiz.rglob(nome)) | set(raiz.rglob(nome + ".gz")))
    encontrados = [p for p in encontrados if p.is_file()]
    if len(encontrados) == 1:
        return encontrados[0]
    if not encontrados:
        raise FileNotFoundError(
            f"Falta {nome} (ou {nome}.gz) em {raiz.resolve()}. "
            "Extraia o ZIP do download; os arquivos .csv.gz podem permanecer comprimidos."
        )
    raise ValueError(f"Mais de uma cópia de {nome}; mantenha uma versão: {encontrados}")
