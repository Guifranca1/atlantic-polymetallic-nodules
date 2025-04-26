# src/data/setup_database.py

import os
import sqlite3
import pandas as pd
import sys
import re

def create_database(db_path='data/nodules.db'):
    """
    Cria o banco de dados e configura o esquema.
    
    Parâmetros:
    -----------
    db_path : str
        Caminho para o arquivo do banco de dados SQLite
    
    Retorna:
    --------
    bool
        True se a criação foi bem-sucedida, False caso contrário
    """
    print(f"Criando banco de dados em: {db_path}")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Verificar se o banco de dados já existe
    db_exists = os.path.exists(db_path)
    if db_exists:
        print(f"Aviso: Banco de dados já existe em {db_path}")
        response = input("Deseja sobrescrever o banco existente? (s/n): ")
        if response.lower() != 's':
            print("Operação cancelada pelo usuário.")
            return False
    
    try:
        # Conectar ao banco de dados (cria se não existir)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ler o arquivo de esquema SQL
        try:
            with open('sql/schema.sql', 'r') as f:
                schema_sql = f.read()
        except FileNotFoundError:
            print("Erro: Arquivo sql/schema.sql não encontrado.")
            return False
        
        # Dividir o script SQL em comandos individuais para melhor tratamento de erros
        # Remove comentários e quebras de linha para uma melhor divisão
        schema_sql = re.sub(r'--.*?\n', '\n', schema_sql)
        commands = schema_sql.split(';')
        
        # Executar cada comando individualmente
        for i, command in enumerate(commands):
            command = command.strip()
            if command:  # Ignorar comandos vazios
                try:
                    cursor.execute(command + ';')
                    print(f"Comando SQL {i+1}/{len(commands)} executado com sucesso")
                except sqlite3.Error as e:
                    print(f"Erro ao executar comando SQL {i+1}: {e}")
                    print(f"Comando problemático: {command}")
                    # Perguntar se deve continuar
                    response = input("Continuar apesar do erro? (s/n): ")
                    if response.lower() != 's':
                        conn.close()
                        return False
        
        # Commit e fechar conexão
        conn.commit()
        conn.close()
        
        print("Banco de dados criado com sucesso!")
        return True
        
    except sqlite3.Error as e:
        print(f"Erro SQLite ao criar banco de dados: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado ao criar banco de dados: {e}")
        return False

def insert_sample_data(db_path='data/nodules.db'):
    """
    Insere dados de amostra no banco de dados para teste.
    
    Parâmetros:
    -----------
    db_path : str
        Caminho para o arquivo do banco de dados SQLite
    
    Retorna:
    --------
    dict
        Resumo da inserção
    """
    print("Inserindo dados de amostra no banco de dados...")
    
    # Verificar se o banco de dados existe
    if not os.path.exists(db_path):
        print(f"Erro: Banco de dados não encontrado em {db_path}")
        return {'error': 'Database not found'}
    
    summary = {}
    conn = None
    
    try:
        # Conectar ao banco de dados
        conn = sqlite3.connect(db_path)
        
        # Verificar se as tabelas necessárias existem
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['nodule_samples', 'metal_prices', 'ports', 
                          'exploration_contracts', 'ocean_conditions']
        
        missing_tables = [table for table in required_tables if table not in tables]
        if missing_tables:
            print(f"Erro: Tabelas necessárias não encontradas: {', '.join(missing_tables)}")
            return {'error': 'Missing tables'}
        
        # 1. Inserir amostras de nódulos
        try:
            nodule_samples = [
                {
                    'sample_id': 'ATL001',
                    'latitude': -15.5,
                    'longitude': -25.3,
                    'depth_m': 4500.0,
                    'date_collected': '2020-06-15',
                    'source': 'ISA Research Expedition',
                    'collection_method': 'Box corer',
                    'mn_pct': 24.5,
                    'ni_pct': 1.3,
                    'cu_pct': 1.1,
                    'co_pct': 0.25,
                    'fe_pct': 6.2,
                    'si_pct': 8.1,
                    'al_pct': 4.2,
                    'density_kg_m2': 11.2,
                    'nodule_size_mm': 45.0,
                    'notes': 'High grade sample from Rio Grande Rise'
                },
                {
                    'sample_id': 'ATL002',
                    'latitude': -12.8,
                    'longitude': -28.9,
                    'depth_m': 4680.0,
                    'date_collected': '2020-06-18',
                    'source': 'ISA Research Expedition',
                    'collection_method': 'Box corer',
                    'mn_pct': 22.1,
                    'ni_pct': 1.2,
                    'cu_pct': 0.9,
                    'co_pct': 0.22,
                    'fe_pct': 5.9,
                    'si_pct': 9.2,
                    'al_pct': 4.5,
                    'density_kg_m2': 9.8,
                    'nodule_size_mm': 38.0,
                    'notes': 'Medium grade sample from Rio Grande Rise'
                }
            ]
            
            nodules_df = pd.DataFrame(nodule_samples)
            nodules_df.to_sql('nodule_samples', conn, if_exists='append', index=False)
            summary['nodule_samples'] = len(nodules_df)
            print(f"Inseridos {len(nodules_df)} registros na tabela 'nodule_samples'")
        except sqlite3.Error as e:
            print(f"Erro ao inserir dados em 'nodule_samples': {e}")
            summary['nodule_samples'] = {'error': str(e)}
        
        # 2. Inserir preços de metais
        try:
            metal_prices = [
                {
                    'metal_name': 'Nickel',
                    'date': '2023-01-01',
                    'price_usd_ton': 15000.0,
                    'source': 'LME'
                },
                {
                    'metal_name': 'Copper',
                    'date': '2023-01-01',
                    'price_usd_ton': 8000.0,
                    'source': 'LME'
                },
                {
                    'metal_name': 'Cobalt',
                    'date': '2023-01-01',
                    'price_usd_ton': 30000.0,
                    'source': 'LME'
                },
                {
                    'metal_name': 'Manganese',
                    'date': '2023-01-01',
                    'price_usd_ton': 1700.0,
                    'source': 'LME'
                }
            ]
            
            prices_df = pd.DataFrame(metal_prices)
            prices_df.to_sql('metal_prices', conn, if_exists='append', index=False)
            summary['metal_prices'] = len(prices_df)
            print(f"Inseridos {len(prices_df)} registros na tabela 'metal_prices'")
        except sqlite3.Error as e:
            print(f"Erro ao inserir dados em 'metal_prices': {e}")
            summary['metal_prices'] = {'error': str(e)}
        
        # 3. Inserir portos
        try:
            ports = [
                {
                    'port_id': 'BRA001',
                    'name': 'Rio de Janeiro',
                    'country': 'Brazil',
                    'latitude': -22.8944,
                    'longitude': -43.1729,
                    'max_depth_m': 15.0,
                    'has_mineral_processing': True,
                    'annual_capacity_tons': 5000000
                },
                {
                    'port_id': 'USA001',
                    'name': 'Charleston',
                    'country': 'USA',
                    'latitude': 32.7833,
                    'longitude': -79.9333,
                    'max_depth_m': 14.0,
                    'has_mineral_processing': False,
                    'annual_capacity_tons': 9000000
                }
            ]
            
            ports_df = pd.DataFrame(ports)
            ports_df.to_sql('ports', conn, if_exists='append', index=False)
            summary['ports'] = len(ports_df)
            print(f"Inseridos {len(ports_df)} registros na tabela 'ports'")
        except sqlite3.Error as e:
            print(f"Erro ao inserir dados em 'ports': {e}")
            summary['ports'] = {'error': str(e)}
        
        # 4. Inserir contratos de exploração
        try:
            contracts = [
                {
                    'contract_id': 'ISA-ATL-001',
                    'contractor': 'Federal Institute for Geosciences and Natural Resources of Germany',
                    'sponsoring_state': 'Germany',
                    'resource_type': 'Polymetallic Nodules',
                    'location': 'Atlantic Ocean',
                    'area_km2': 77230.0,
                    'date_signed': '2006-07-19',
                    'expires': '2026-07-18'
                },
                {
                    'contract_id': 'ISA-ATL-002',
                    'contractor': 'UK Seabed Resources Ltd',
                    'sponsoring_state': 'United Kingdom',
                    'resource_type': 'Polymetallic Nodules',
                    'location': 'Atlantic Ocean',
                    'area_km2': 74500.0,
                    'date_signed': '2016-03-29',
                    'expires': '2031-03-28'
                }
            ]
            
            contracts_df = pd.DataFrame(contracts)
            contracts_df.to_sql('exploration_contracts', conn, if_exists='append', index=False)
            summary['exploration_contracts'] = len(contracts_df)
            print(f"Inseridos {len(contracts_df)} registros na tabela 'exploration_contracts'")
        except sqlite3.Error as e:
            print(f"Erro ao inserir dados em 'exploration_contracts': {e}")
            summary['exploration_contracts'] = {'error': str(e)}
        
        # 5. Inserir condições oceânicas
        try:
            ocean_conditions = [
                {
                    'latitude': -15.5,
                    'longitude': -25.3,
                    'date': '2023-01-01',
                    'current_velocity_ms': 0.15,
                    'temperature_c': 2.3,
                    'salinity_psu': 34.8,
                    'oxygen_ml_l': 4.2,
                    'source': 'World Ocean Atlas'
                },
                {
                    'latitude': -12.8,
                    'longitude': -28.9,
                    'date': '2023-01-01',
                    'current_velocity_ms': 0.18,
                    'temperature_c': 2.5,
                    'salinity_psu': 34.9,
                    'oxygen_ml_l': 4.0,
                    'source': 'World Ocean Atlas'
                }
            ]
            
            conditions_df = pd.DataFrame(ocean_conditions)
            conditions_df.to_sql('ocean_conditions', conn, if_exists='append', index=False)
            summary['ocean_conditions'] = len(conditions_df)
            print(f"Inseridos {len(conditions_df)} registros na tabela 'ocean_conditions'")
        except sqlite3.Error as e:
            print(f"Erro ao inserir dados em 'ocean_conditions': {e}")
            summary['ocean_conditions'] = {'error': str(e)}
        
        # Commit alterações
        conn.commit()
        print("Dados de amostra inseridos com sucesso!")
        
    except sqlite3.Error as e:
        print(f"Erro SQLite ao inserir dados: {e}")
        summary['error'] = str(e)
    except Exception as e:
        print(f"Erro inesperado ao inserir dados: {e}")
        summary['error'] = str(e)
    finally:
        # Garantir que a conexão seja fechada mesmo em caso de erro
        if conn:
            conn.close()
        
    return summary

def verify_database(db_path='data/nodules.db'):
    """
    Verifica a integridade do banco de dados.
    
    Parâmetros:
    -----------
    db_path : str
        Caminho para o arquivo do banco de dados SQLite
    
    Retorna:
    --------
    bool
        True se o banco de dados está íntegro, False caso contrário
    """
    print(f"Verificando integridade do banco de dados em: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Erro: Banco de dados não encontrado em {db_path}")
        return False
    
    try:
        # Conectar ao banco de dados
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar integridade do banco de dados
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        
        # Verificar tabelas existentes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
        
        # Verificar contagem de registros em cada tabela
        table_counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_counts[table] = count
        
        # Fechar conexão
        conn.close()
        
        # Exibir resultados
        print(f"Verificação de integridade: {result}")
        print("Tabelas encontradas:")
        for table, count in table_counts.items():
            print(f"  - {table}: {count} registros")
        
        return result == "ok"
        
    except sqlite3.Error as e:
        print(f"Erro SQLite ao verificar banco de dados: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado ao verificar banco de dados: {e}")
        return False

def export_database_structure(db_path='data/nodules.db', output_file='docs/database_structure.md'):
    """
    Exporta a estrutura do banco de dados para um arquivo Markdown.
    
    Parâmetros:
    -----------
    db_path : str
        Caminho para o arquivo do banco de dados SQLite
    output_file : str
        Caminho para o arquivo de saída Markdown
    
    Retorna:
    --------
    bool
        True se a exportação foi bem-sucedida, False caso contrário
    """
    print(f"Exportando estrutura do banco de dados para: {output_file}")
    
    if not os.path.exists(db_path):
        print(f"Erro: Banco de dados não encontrado em {db_path}")
        return False
    
    try:
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Conectar ao banco de dados
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obter lista de tabelas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Abrir arquivo de saída
        with open(output_file, 'w') as f:
            f.write("# Estrutura do Banco de Dados de Nódulos Polimetálicos\n\n")
            
            # Para cada tabela
            for table in tables:
                f.write(f"## Tabela: `{table}`\n\n")
                
                # Obter informações das colunas
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                
                # Criar tabela Markdown
                f.write("| Coluna | Tipo | Nulo? | Chave | Padrão |\n")
                f.write("|--------|------|-------|-------|--------|\n")
                
                for col in columns:
                    cid, name, type_, notnull, dflt_value, pk = col
                    null_str = "Não" if notnull else "Sim"
                    key_str = "Primária" if pk else ""
                    default_str = dflt_value if dflt_value is not None else ""
                    
                    f.write(f"| {name} | {type_} | {null_str} | {key_str} | {default_str} |\n")
                
                f.write("\n")
                
                # Obter índices
                cursor.execute(f"PRAGMA index_list({table})")
                indices = cursor.fetchall()
                
                if indices:
                    f.write("### Índices\n\n")
                    f.write("| Nome | Colunas | Único? |\n")
                    f.write("|------|---------|--------|\n")
                    
                    for idx in indices:
                        idx_name = idx[1]
                        is_unique = "Sim" if idx[2] else "Não"
                        
                        # Obter colunas do índice
                        cursor.execute(f"PRAGMA index_info({idx_name})")
                        idx_columns = cursor.fetchall()
                        columns_str = ", ".join([table_info[2] for table_info in idx_columns])
                        
                        f.write(f"| {idx_name} | {columns_str} | {is_unique} |\n")
                
                f.write("\n")
                
                # Obter contagem de registros
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                f.write(f"Total de registros: {count}\n\n")
                
                # Amostra de dados
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 5")
                    rows = cursor.fetchall()
                    
                    f.write("### Amostra de Dados\n\n")
                    
                    # Cabeçalho da tabela
                    header = "|"
                    separator = "|"
                    for col in columns:
                        header += f" {col[1]} |"
                        separator += " --- |"
                    
                    f.write(header + "\n")
                    f.write(separator + "\n")
                    
                    # Dados
                    for row in rows:
                        row_str = "|"
                        for val in row:
                            val_str = str(val) if val is not None else "NULL"
                            row_str += f" {val_str} |"
                        f.write(row_str + "\n")
                    
                    f.write("\n")
        
        # Fechar conexão
        conn.close()
        
        print(f"Estrutura do banco de dados exportada com sucesso para {output_file}")
        return True
        
    except sqlite3.Error as e:
        print(f"Erro SQLite ao exportar estrutura: {e}")
        return False
    except Exception as e:
        print(f"Erro inesperado ao exportar estrutura: {e}")
        return False

def main():
    """
    Função principal para configurar o banco de dados.
    """
    db_path = 'data/nodules.db'
    
    # Verificar argumentos da linha de comando
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'create':
            # Criar banco de dados
            create_database(db_path)
        elif command == 'sample':
            # Inserir dados de amostra
            insert_sample_data(db_path)
        elif command == 'verify':
            # Verificar banco de dados
            verify_database(db_path)
        elif command == 'export':
            # Exportar estrutura
            export_database_structure(db_path)
        elif command == 'all':
            # Executar todas as operações
            if create_database(db_path):
                insert_sample_data(db_path)
                verify_database(db_path)
                export_database_structure(db_path)
        else:
            print(f"Comando desconhecido: {command}")
            print("Comandos disponíveis: create, sample, verify, export, all")
    else:
        # Sem argumentos, executar fluxo padrão
        print("Iniciando configuração completa do banco de dados...")
        
        # Criar banco de dados
        if create_database(db_path):
            # Inserir dados de amostra
            summary = insert_sample_data(db_path)
            
            print("\nResumo da configuração do banco de dados:")
            for table, info in summary.items():
                if isinstance(info, dict) and 'error' in info:
                    print(f"  {table}: ERRO - {info['error']}")
                else:
                    print(f"  {table}: {info} registros")
            
            # Verificar integridade do banco de dados
            verify_database(db_path)
            
            # Exportar estrutura para documentação
            export_database_structure(db_path)
    
if __name__ == "__main__":
    main()