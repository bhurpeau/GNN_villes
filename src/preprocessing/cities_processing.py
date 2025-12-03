import pandas as pd
from io import load_parquet_data, load_csv_data, save_parquet_data


def process_macro_flows(dad_path, dt_path, pop_path, acti_path, output_nodes, output_edges):
    dad = load_parquet_data(dad_path)
    dt = load_parquet_data(dt_path)
    pop = load_csv_data(pop_path)
    acti = load_csv_data(acti_path)
    dad.columns =['pop21', 'code_a', 'code']
    dt.columns = ['pop21', 'code', 'code_a']

    dad_edges=dad.groupby(['code', 'code_a']).pop21.sum()
    dad_edges = dad_edges.reset_index()

    dt_edges=dt.groupby(['code', 'code_a']).pop21.sum()
    dt_edges = dt_edges.reset_index()

    dad_edges_final = dad_edges.loc[dad_edges['pop21']>50].copy()
    dt_edges_final = dt_edges.loc[dt_edges['pop21']>50].copy()

    pop = pop[['COM', 'PMUN']]
    pop.columns = ['code', 'pop']
    dad_edges_final = dad_edges_final.merge(pop, how='left', on='code')
    dad_edges_final['migra'] = dad_edges_final.apply(lambda x: float(x['pop21'])/x['pop'], axis=1)
    dad_edges_final = dad_edges_final[['code', 'code_a', 'migra']]

    acti = acti[['CODGEO', 'P21_POP1564']]
    acti.columns = ['code', 'pop']
    dt_edges_final = dt_edges_final.merge(acti, how='left', on='code')
    dt_edges_final['d_t'] = dt_edges_final.apply(lambda x: float(x['pop21'])/x['pop'], axis=1)
    dt_edges_final = dt_edges_final[['code', 'code_a', 'd_t']]

    dt_edges_final = dt_edges_final.loc[~dt_edges_final['code_a'].isin(['YYYYY', '99999'])].copy()
    dad_edges_final = dad_edges_final.loc[~dad_edges_final['code_a'].isin(['YYYYY', '99999'])].copy()
    edges_final = dt_edges_final.merge(dad_edges_final, how='outer', on=['code', 'code_a']).fillna(0)
    # 1. Extraction des flux internes (Diagonale de la matrice)
    # Ce sont les gens qui vivent et travaillent/restent dans la même commune
    flux_internes = edges_final[edges_final['code'] == edges_final['code_a']].copy()

    # On renomme pour que ce soit clair
    flux_internes = flux_internes.rename(columns={
        'd_t': 'macro_taux_retenue',   # Part des actifs qui restent (Autonomie)
        'migra': 'macro_taux_stabilite' # Part des gens qui ne déménagent pas (Inverse du Turnover)
    })

    save_parquet_data(flux_internes[['code', 'macro_taux_retenue', 'macro_taux_stabilite']], output_nodes)
    # 2. Nettoyage des ARÊTES
    # Pour le graphe, on ne garde que les flux vers les AUTRES communes
    edges_final = edges_final[edges_final['code'] != edges_final['code_a']]

    # 3. Sauvegarde finale
    save_parquet_data(edges_final, output_edges)
    return flux_internes, edges_final

