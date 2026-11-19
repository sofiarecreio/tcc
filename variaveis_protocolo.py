"""Contrato dos dois experimentos; o painel completo é o principal."""
COMUM_NUM = ['idade', 'fc_24h', 'fr_24h', 'spo2_24h', 'pam_24h', 'pas_24h',
             'creatinina_24h', 'bun_24h', 'sodio_24h', 'potassio_24h', 'glicose_24h', 'ptt_24h']
COMPLETO_NUM = [
    'idade', 'pam_24h', 'fc_24h', 'fr_24h', 'spo2_24h', 'creatinina_24h',
    'ureia_24h', 'sodio_24h', 'potassio_24h', 'glicose_24h', 'ptt_24h', 'feve',
    'temperatura_mediana_24h', 'gcs_min_24h', 'fio2_max_24h', 'peep_max_24h',
    'lactato_max_24h', 'bicarbonato_min_24h', 'leucocitos_max_24h', 'hemoglobina_min_24h',
    'plaquetas_min_24h', 'inr_max_24h', 'bilirrubina_max_24h', 'nt_probnp_max_24h',
    'troponina_t_max_24h', 'ph_min_24h', 'pco2_max_24h',
    'noradrenalina_taxa_max_24h', 'n_vasopressores_24h', 'furosemida_dose_total_24h',
    'volume_infundido_24h', 'diurese_total_24h', 'diurese_ml_kg_h_24h',
    'balanco_hidrico_24h', 'balanco_ml_kg_24h']
COMPLETO_BIN = [
    'registro_ventilador_24h', 'lactato_medido', 'nt_probnp_medido', 'troponina_medida',
    'gasometria_medida', 'vasopressor_24h', 'noradrenalina_24h', 'vasopressina_24h',
    'adrenalina_24h', 'fenilefrina_24h', 'dopamina_24h', 'inotropico_24h',
    'diuretico_alca_24h', 'furosemida_infusao_continua_24h', 'bolus_cristaloide_24h',
    'oliguria_24h', 'diurese_medida']
PERFIS = {
    'completo': {'numericas': COMPLETO_NUM, 'binarias': COMPLETO_BIN,
                 'categoricas': ['genero', 'tipo_admissao']},
    'comum': {'numericas': COMUM_NUM, 'binarias': [], 'categoricas': ['genero']},
}


def colunas(perfil):
    p = PERFIS[perfil]
    return p['numericas'] + p['binarias'] + p['categoricas']
