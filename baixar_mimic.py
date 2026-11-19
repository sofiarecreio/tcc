"""Abre o cliente oficial no terminal; a senha é solicitada pelo próprio PhysioNet."""
import json
import os
from pathlib import Path
import runpy
import sys

RAIZ = Path(__file__).resolve().parent


def main():
    os.chdir(RAIZ)
    cliente = RAIZ / 'tmp' / 'physionet_client'
    if cliente.exists():
        sys.path.insert(0, str(cliente))
    destino = RAIZ / 'data' / 'mimic-iv'
    destino.mkdir(parents=True, exist_ok=True)
    estado = destino / 'download_status.json'
    def registrar(status, codigo=None):
        estado.write_text(json.dumps({'status': status, 'codigo_saida': codigo,
            'destino': str(destino), 'base': 'mimiciv', 'versao': '3.1'}, indent=2), encoding='utf-8')
    print('MIMIC-IV 3.1 - download pelo cliente oficial do PhysioNet')
    print('Destino:', destino)
    print('Digite sua senha SOMENTE nesta janela quando Password for solicitado.')
    print('A senha nao aparece enquanto voce digita. Pressione Enter ao terminar.')
    print('Mantenha esta janela aberta durante o download.\n', flush=True)
    sys.argv = ['physionet', 'download', 'mimiciv', '--version', '3.1',
                '--source', 'physionet', '--username', 'sofiarecreiouff',
                '--output', str(destino)]
    registrar('cliente_iniciado_aguardando_login_ou_download')
    codigo = 0
    try:
        runpy.run_module('physionet', run_name='__main__')
    except SystemExit as exc:
        codigo = exc.code or 0
    except KeyboardInterrupt:
        codigo = 130
    except Exception as exc:
        print('Falha no cliente:', type(exc).__name__)
        codigo = 1
    registrar('concluido' if codigo == 0 else 'interrompido_ou_erro', codigo)
    print('\nDownload concluido.' if codigo == 0 else '\nDownload nao concluido. Confira a mensagem acima.')
    input('Pressione Enter para fechar esta janela...')


if __name__ == '__main__':
    main()
