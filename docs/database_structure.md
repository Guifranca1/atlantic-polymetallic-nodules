# Estrutura do Banco de Dados de Nódulos Polimetálicos

## Tabela: `nodule_samples`

| Coluna | Tipo | Nulo? | Chave | Padrão |
|--------|------|-------|-------|--------|
| sample_id | VARCHAR(50) | Sim | Primária |  |
| latitude | DECIMAL(10,6) | Não |  |  |
| longitude | DECIMAL(10,6) | Não |  |  |
| depth_m | DECIMAL(8,2) | Sim |  |  |
| date_collected | DATE | Sim |  |  |
| source | VARCHAR(100) | Sim |  |  |
| collection_method | VARCHAR(100) | Sim |  |  |
| mn_pct | DECIMAL(5,2) | Sim |  |  |
| ni_pct | DECIMAL(5,2) | Sim |  |  |
| cu_pct | DECIMAL(5,2) | Sim |  |  |
| co_pct | DECIMAL(5,2) | Sim |  |  |
| fe_pct | DECIMAL(5,2) | Sim |  |  |
| si_pct | DECIMAL(5,2) | Sim |  |  |
| al_pct | DECIMAL(5,2) | Sim |  |  |
| density_kg_m2 | DECIMAL(8,2) | Sim |  |  |
| nodule_size_mm | DECIMAL(6,2) | Sim |  |  |
| notes | TEXT | Sim |  |  |
| created_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |

### Índices

| Nome | Colunas | Único? |
|------|---------|--------|
| idx_nodule_location | latitude, longitude | Não |
| sqlite_autoindex_nodule_samples_1 | sample_id | Sim |

Total de registros: 2

### Amostra de Dados

| sample_id | latitude | longitude | depth_m | date_collected | source | collection_method | mn_pct | ni_pct | cu_pct | co_pct | fe_pct | si_pct | al_pct | density_kg_m2 | nodule_size_mm | notes | created_at | updated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ATL001 | -15.5 | -25.3 | 4500 | 2020-06-15 | ISA Research Expedition | Box corer | 24.5 | 1.3 | 1.1 | 0.25 | 6.2 | 8.1 | 4.2 | 11.2 | 45 | High grade sample from Rio Grande Rise | 2025-04-26 17:13:05 | 2025-04-26 17:13:05 |
| ATL002 | -12.8 | -28.9 | 4680 | 2020-06-18 | ISA Research Expedition | Box corer | 22.1 | 1.2 | 0.9 | 0.22 | 5.9 | 9.2 | 4.5 | 9.8 | 38 | Medium grade sample from Rio Grande Rise | 2025-04-26 17:13:05 | 2025-04-26 17:13:05 |

## Tabela: `metal_prices`

| Coluna | Tipo | Nulo? | Chave | Padrão |
|--------|------|-------|-------|--------|
| price_id | INTEGER | Sim | Primária |  |
| metal_name | VARCHAR(50) | Não |  |  |
| date | DATE | Não |  |  |
| price_usd_ton | DECIMAL(10,2) | Não |  |  |
| source | VARCHAR(100) | Sim |  |  |
| created_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |

### Índices

| Nome | Colunas | Único? |
|------|---------|--------|
| idx_metal_prices_date | date | Não |
| sqlite_autoindex_metal_prices_1 | metal_name, date | Sim |

Total de registros: 13904

### Amostra de Dados

| price_id | metal_name | date | price_usd_ton | source | created_at |
| --- | --- | --- | --- | --- | --- |
| 1 | Nickel | 2023-01-01 | 15000 | LME | 2025-04-26 17:13:05 |
| 2 | Copper | 2023-01-01 | 8000 | LME | 2025-04-26 17:13:05 |
| 3 | Cobalt | 2023-01-01 | 30000 | LME | 2025-04-26 17:13:05 |
| 4 | Manganese | 2023-01-01 | 1700 | LME | 2025-04-26 17:13:05 |
| 5 | Nickel | 2012-01-02 | 16095.180469236198 | Synthetic Data | 2025-04-26 17:27:39 |

## Tabela: `ports`

| Coluna | Tipo | Nulo? | Chave | Padrão |
|--------|------|-------|-------|--------|
| port_id | VARCHAR(20) | Sim | Primária |  |
| name | VARCHAR(100) | Não |  |  |
| country | VARCHAR(50) | Não |  |  |
| latitude | DECIMAL(10,6) | Não |  |  |
| longitude | DECIMAL(10,6) | Não |  |  |
| max_depth_m | DECIMAL(6,2) | Sim |  |  |
| has_mineral_processing | BOOLEAN | Sim |  | FALSE |
| annual_capacity_tons | INT | Sim |  |  |
| created_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |

### Índices

| Nome | Colunas | Único? |
|------|---------|--------|
| sqlite_autoindex_ports_1 | port_id | Sim |

Total de registros: 2

### Amostra de Dados

| port_id | name | country | latitude | longitude | max_depth_m | has_mineral_processing | annual_capacity_tons | created_at | updated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BRA001 | Rio de Janeiro | Brazil | -22.8944 | -43.1729 | 15 | 1 | 5000000 | 2025-04-26 17:13:05 | 2025-04-26 17:13:05 |
| USA001 | Charleston | USA | 32.7833 | -79.9333 | 14 | 0 | 9000000 | 2025-04-26 17:13:05 | 2025-04-26 17:13:05 |

## Tabela: `exploration_contracts`

| Coluna | Tipo | Nulo? | Chave | Padrão |
|--------|------|-------|-------|--------|
| contract_id | VARCHAR(50) | Sim | Primária |  |
| contractor | VARCHAR(100) | Não |  |  |
| sponsoring_state | VARCHAR(50) | Não |  |  |
| resource_type | VARCHAR(50) | Não |  |  |
| location | VARCHAR(100) | Não |  |  |
| area_km2 | DECIMAL(10,2) | Sim |  |  |
| date_signed | DATE | Sim |  |  |
| expires | DATE | Sim |  |  |
| created_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |
| updated_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |

### Índices

| Nome | Colunas | Único? |
|------|---------|--------|
| sqlite_autoindex_exploration_contracts_1 | contract_id | Sim |

Total de registros: 2

### Amostra de Dados

| contract_id | contractor | sponsoring_state | resource_type | location | area_km2 | date_signed | expires | created_at | updated_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISA-ATL-001 | Federal Institute for Geosciences and Natural Resources of Germany | Germany | Polymetallic Nodules | Atlantic Ocean | 77230 | 2006-07-19 | 2026-07-18 | 2025-04-26 17:13:05 | 2025-04-26 17:13:05 |
| ISA-ATL-002 | UK Seabed Resources Ltd | United Kingdom | Polymetallic Nodules | Atlantic Ocean | 74500 | 2016-03-29 | 2031-03-28 | 2025-04-26 17:13:05 | 2025-04-26 17:13:05 |

## Tabela: `ocean_conditions`

| Coluna | Tipo | Nulo? | Chave | Padrão |
|--------|------|-------|-------|--------|
| condition_id | INTEGER | Sim | Primária |  |
| latitude | DECIMAL(10,6) | Não |  |  |
| longitude | DECIMAL(10,6) | Não |  |  |
| date | DATE | Não |  |  |
| current_velocity_ms | DECIMAL(6,3) | Sim |  |  |
| temperature_c | DECIMAL(5,2) | Sim |  |  |
| salinity_psu | DECIMAL(5,2) | Sim |  |  |
| oxygen_ml_l | DECIMAL(5,2) | Sim |  |  |
| source | VARCHAR(100) | Sim |  |  |
| created_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |

### Índices

| Nome | Colunas | Único? |
|------|---------|--------|
| idx_ocean_conditions_location | latitude, longitude | Não |
| sqlite_autoindex_ocean_conditions_1 | latitude, longitude, date | Sim |

Total de registros: 2

### Amostra de Dados

| condition_id | latitude | longitude | date | current_velocity_ms | temperature_c | salinity_psu | oxygen_ml_l | source | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -15.5 | -25.3 | 2023-01-01 | 0.15 | 2.3 | 34.8 | 4.2 | World Ocean Atlas | 2025-04-26 17:13:05 |
| 2 | -12.8 | -28.9 | 2023-01-01 | 0.18 | 2.5 | 34.9 | 4 | World Ocean Atlas | 2025-04-26 17:13:05 |

## Tabela: `bathymetry`

| Coluna | Tipo | Nulo? | Chave | Padrão |
|--------|------|-------|-------|--------|
| bathymetry_id | INTEGER | Sim | Primária |  |
| latitude | DECIMAL(10,6) | Não |  |  |
| longitude | DECIMAL(10,6) | Não |  |  |
| depth_m | DECIMAL(8,2) | Não |  |  |
| source | VARCHAR(100) | Sim |  |  |
| created_at | TIMESTAMP | Sim |  | CURRENT_TIMESTAMP |

### Índices

| Nome | Colunas | Único? |
|------|---------|--------|
| idx_bathymetry_location | latitude, longitude | Não |
| sqlite_autoindex_bathymetry_1 | latitude, longitude | Sim |

Total de registros: 41208

### Amostra de Dados

| bathymetry_id | latitude | longitude | depth_m | source | created_at |
| --- | --- | --- | --- | --- | --- |
| 1 | -15 | -45 | 4091.556798923328 | mid-atlantic_ridge_bathymetry.csv | 2025-04-26 14:25:10 |
| 2 | -15 | -44.9 | 4340.718998390745 | mid-atlantic_ridge_bathymetry.csv | 2025-04-26 14:25:10 |
| 3 | -15 | -44.8 | 3720.705602030272 | mid-atlantic_ridge_bathymetry.csv | 2025-04-26 14:25:10 |
| 4 | -15 | -44.7 | 4047.809653724568 | mid-atlantic_ridge_bathymetry.csv | 2025-04-26 14:25:10 |
| 5 | -15 | -44.6 | 3734.301391680469 | mid-atlantic_ridge_bathymetry.csv | 2025-04-26 14:25:10 |

