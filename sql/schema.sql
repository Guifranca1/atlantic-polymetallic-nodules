-- sql/schema.sql

-- Schema for Atlantic Polymetallic Nodules Database

-- Nodule Samples Table
CREATE TABLE nodule_samples (
    sample_id VARCHAR(50) PRIMARY KEY,
    latitude DECIMAL(10,6) NOT NULL,
    longitude DECIMAL(10,6) NOT NULL,
    depth_m DECIMAL(8,2),
    date_collected DATE,
    source VARCHAR(100),
    collection_method VARCHAR(100),
    
    -- Chemical composition (percentage)
    mn_pct DECIMAL(5,2),
    ni_pct DECIMAL(5,2),
    cu_pct DECIMAL(5,2),
    co_pct DECIMAL(5,2),
    fe_pct DECIMAL(5,2),
    si_pct DECIMAL(5,2),
    al_pct DECIMAL(5,2),
    
    -- Physical properties
    density_kg_m2 DECIMAL(8,2),
    nodule_size_mm DECIMAL(6,2),
    
    -- Metadata
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Metal Prices Table
CREATE TABLE metal_prices (
    price_id INT AUTO_INCREMENT PRIMARY KEY,
    metal_name VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    price_usd_ton DECIMAL(10,2) NOT NULL,
    source VARCHAR(100),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY (metal_name, date)
);

-- Ports Table
CREATE TABLE ports (
    port_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    country VARCHAR(50) NOT NULL,
    latitude DECIMAL(10,6) NOT NULL,
    longitude DECIMAL(10,6) NOT NULL,
    
    -- Port specifications
    max_depth_m DECIMAL(6,2),
    has_mineral_processing BOOLEAN DEFAULT FALSE,
    annual_capacity_tons INT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Exploration Contracts Table
CREATE TABLE exploration_contracts (
    contract_id VARCHAR(50) PRIMARY KEY,
    contractor VARCHAR(100) NOT NULL,
    sponsoring_state VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    location VARCHAR(100) NOT NULL,
    area_km2 DECIMAL(10,2),
    date_signed DATE,
    expires DATE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Ocean Conditions Table
CREATE TABLE ocean_conditions (
    condition_id INT AUTO_INCREMENT PRIMARY KEY,
    latitude DECIMAL(10,6) NOT NULL,
    longitude DECIMAL(10,6) NOT NULL,
    date DATE NOT NULL,
    
    -- Conditions
    current_velocity_ms DECIMAL(6,3),
    temperature_c DECIMAL(5,2),
    salinity_psu DECIMAL(5,2),
    oxygen_ml_l DECIMAL(5,2),
    
    -- Metadata
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE KEY (latitude, longitude, date)
);

-- Create indices for efficient queries
CREATE INDEX idx_nodule_location ON nodule_samples(latitude, longitude);
CREATE INDEX idx_metal_prices_date ON metal_prices(date);
CREATE INDEX idx_ocean_conditions_location ON ocean_conditions(latitude, longitude);