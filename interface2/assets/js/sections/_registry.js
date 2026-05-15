/* ─────────────────────────────────────────────────────────────────────────
   SECTIONS REGISTRY — 16 secciones tipo Monitor PM (S1-S16) + Overview.
   Cada chart declara su dataSource (catalog real o mock con seed).
   ───────────────────────────────────────────────────────────────────── */
"use strict";

window.BCCh = window.BCCh || {};

const catalog = (query_id, opts = {}) => ({ type: "catalog", query_id, ...opts });
const mock    = (seed = 42, opts = {}) => ({ type: "mock", seed, ...opts });

// ── AGENTE GOEM (chat embebido) ──────────────────────────────────────────

const AGENTE_GOEM = {
    id: "agente_goem",
    title: "Agente GOEM",
    subtitle: "Conversa con el agente sobre el corpus BCCh y el catálogo SQL",
    group: "Inicio",
    kind: "chat",
    charts: [],
};

// ── MERCADOS (terminal Bloomberg-dense) ─────────────────────────────────

const MERCADOS = {
    id: "mercados",
    title: "Mercados",
    subtitle: "Monitor denso de mercados Chile y externos · 6 paneles",
    group: "Inicio",
    kind: "markets",
    charts: [],
};

// ── MERCADOS EN LÍNEA (vista retail estilo LarrainVial) ─────────────────

const MERCADOS_ONLINE = {
    id: "mercados_online",
    title: "Mercados en Línea",
    subtitle: "Índices, acciones, monedas, commodities, tasas e indicadores",
    group: "Inicio",
    kind: "markets-online",
    charts: [],
};

// ── OVERVIEW ─────────────────────────────────────────────────────────────

const OVERVIEW = {
    id: "overview",
    title: "Resumen General",
    subtitle: "Dashboard general · principales indicadores del día",
    group: "Inicio",
    charts: [
        {
            id: "ov_usdclp",
            title: "USD/CLP — Últimos 12 meses",
            type: "line",
            dataSource: catalog("usdclp_historico"),
            xField: "fecha", yFields: ["usdclp"],
            unit: "CLP/USD",
            defaultRange: "1Y",
            temporality: ["1M", "3M", "6M", "1Y", "YTD", "Max"],
        },
        {
            id: "ov_cobre",
            title: "Precio del Cobre",
            type: "line",
            dataSource: catalog("precio_cobre"),
            xField: "fecha",
            unit: "USD/lb",
            defaultRange: "1Y",
            temporality: ["1M", "3M", "6M", "1Y", "YTD", "Max"],
        },
        {
            id: "ov_btp_curva",
            title: "Curva BTP CLP",
            type: "line",
            dataSource: catalog("curva_btp_clp"),
            xField: "fecha",
            unit: "% nominal",
            yFormat: "percent",
            defaultRange: "6M",
        },
        {
            id: "ov_tpm_exp",
            title: "Expectativas TPM implícita",
            type: "line",
            dataSource: catalog("expectativas_tpm_mipr"),
            xField: "fecha",
            unit: "pb",
            yFormat: "bps",
            defaultRange: "3M",
        },
    ],
};

// ── MERCADO ──────────────────────────────────────────────────────────────

const S1 = {
    id: "s1", title: "Mercado Monetario", group: "Mercado",
    subtitle: "Tasas DAP, swap, PDBC, TIB",
    charts: [
        { id: "s1_dapswap1m",  title: "Spread DAP-Swap 1M",   type: "line",
          dataSource: catalog("spreads_dap_swap_clp"), xField: "fecha", yFields: ["dap_swap_1m"],
          unit: "pb", yFormat: "bps", defaultRange: "6M" },
        { id: "s1_dapswap3m",  title: "Spread DAP-Swap 3M",   type: "line",
          dataSource: catalog("spreads_dap_swap_clp"), xField: "fecha", yFields: ["dap_swap_3m"],
          unit: "pb", yFormat: "bps", defaultRange: "6M" },
        { id: "s1_dapswap6m",  title: "Spread DAP-Swap 6M",   type: "line",
          dataSource: catalog("spreads_dap_swap_clp"), xField: "fecha", yFields: ["dap_swap_6m"],
          unit: "pb", yFormat: "bps", defaultRange: "6M" },
        { id: "s1_dapswap12m", title: "Spread DAP-Swap 12M",  type: "line",
          dataSource: catalog("spreads_dap_swap_clp"), xField: "fecha", yFields: ["dap_swap_12m"],
          unit: "pb", yFormat: "bps", defaultRange: "6M" },
        { id: "s1_pdbc",       title: "Tasas PDBC Mercado Secundario", type: "line",
          dataSource: catalog("pdbc_bolsa"), xField: "fecha",
          unit: "% nominal", yFormat: "percent", defaultRange: "1Y" },
        { id: "s1_tib",        title: "TIB vs Monto Transado",  type: "line",
          dataSource: catalog("tib_mercado"), xField: "fecha",
          unit: "% / MM CLP", defaultRange: "3M" },
        { id: "s1_dapprime",   title: "Spread DAP/Prime-Swap UF 12M",   type: "line",
          dataSource: mock(101, { mockType: "single" }),
          mockConfig: { start: 25, vol: 0.04 }, unit: "pb", yFormat: "bps" },
        { id: "s1_dapuf6m",    title: "Spread DAP UF 6M",        type: "line",
          dataSource: mock(102), mockConfig: { start: 30, vol: 0.05 }, unit: "pb", yFormat: "bps" },
    ],
};

const S2 = {
    id: "s2", title: "Liquidez MN", group: "Mercado",
    subtitle: "Operaciones de liquidez en moneda nacional",
    charts: [
        { id: "s2_ops",      title: "Operaciones de Liquidez MN",   type: "area",
          dataSource: mock(201), mockConfig: { start: 5000, vol: 0.08 }, unit: "MM CLP" },
        { id: "s2_concentr", title: "Liquidez MN vs Concentración", type: "line",
          dataSource: mock(202, { mockType: "multi", mockSeries: ["Liquidez", "IHH"] }),
          mockConfig: { start: 100, vol: 0.03 }, unit: "Índice" },
        { id: "s2_ratio",    title: "Ratio Liquidez/Obligaciones 30d MN", type: "line",
          dataSource: mock(203), mockConfig: { start: 0.85, vol: 0.02 }, unit: "Ratio" },
        { id: "s2_lcr",      title: "LCR MN", type: "line",
          dataSource: mock(204), mockConfig: { start: 130, vol: 0.025 }, unit: "% LCR" },
    ],
};

const S3 = {
    id: "s3", title: "Liquidez MX", group: "Mercado",
    subtitle: "Liquidez en moneda extranjera",
    charts: [
        { id: "s3_spread1m",  title: "Spread Liquidez MX 1M",  type: "line",
          dataSource: mock(301), mockConfig: { start: 15, vol: 0.06 }, unit: "pb", yFormat: "bps" },
        { id: "s3_spread3m",  title: "Spread Liquidez MX 3M",  type: "line",
          dataSource: mock(302), mockConfig: { start: 22, vol: 0.05 }, unit: "pb", yFormat: "bps" },
        { id: "s3_spread6m",  title: "Spread Liquidez MX 6M",  type: "line",
          dataSource: mock(303), mockConfig: { start: 28, vol: 0.05 }, unit: "pb", yFormat: "bps" },
        { id: "s3_spread12m", title: "Spread Liquidez MX 12M", type: "line",
          dataSource: mock(304), mockConfig: { start: 35, vol: 0.05 }, unit: "pb", yFormat: "bps" },
        { id: "s3_ihh",       title: "Liquidez MX vs Concentración IHH", type: "line",
          dataSource: mock(305, { mockType: "multi", mockSeries: ["Liquidez MX", "IHH"] }),
          unit: "Índice" },
        { id: "s3_ratio",     title: "Ratio Liquidez/Obligaciones 30d", type: "line",
          dataSource: catalog("ratio_liquidez_obligaciones"), xField: "fecha",
          unit: "Ratio", defaultRange: "1Y" },
        { id: "s3_nsfr",      title: "NSFR MX", type: "line",
          dataSource: catalog("nsfr_mx_bancos"), xField: "fecha",
          unit: "% NSFR", defaultRange: "1Y" },
        { id: "s3_lcr",       title: "LCR MX", type: "line",
          dataSource: catalog("lcr_mx_bancos"), xField: "fecha",
          unit: "% LCR", defaultRange: "1Y" },
    ],
};

const S4 = {
    id: "s4", title: "RF · Tasas", group: "Mercado",
    subtitle: "Renta fija — curvas BTP/BTU + UST + spreads",
    charts: [
        { id: "s4_btptu5",     title: "Curva BTP y BTU 5Y", type: "line",
          dataSource: catalog("btp_btu_comparado"), xField: "fecha",
          unit: "% / BEI", defaultRange: "6M" },
        { id: "s4_btp_curva",  title: "Curva BTP completa", type: "line",
          dataSource: catalog("curva_btp_clp"), xField: "fecha",
          unit: "% nominal", yFormat: "percent", defaultRange: "1Y" },
        { id: "s4_btp_ust",    title: "Spread BTP-UST 5Y y 10Y", type: "line",
          dataSource: mock(401, { mockType: "multi", mockSeries: ["5Y", "10Y"] }),
          mockConfig: { start: 80, vol: 0.04 }, unit: "pb", yFormat: "bps" },
        { id: "s4_btp_pend",   title: "Pendiente BTP/BTU 10-5 y 10-2", type: "line",
          dataSource: mock(402, { mockType: "multi", mockSeries: ["10-5 BTP", "10-2 BTP"] }),
          mockConfig: { start: 50, vol: 0.05 }, unit: "pb", yFormat: "bps" },
        { id: "s4_btptu10",    title: "Curva BTP y BTU 10Y", type: "line",
          dataSource: catalog("btp_btu_comparado"), xField: "fecha",
          unit: "% / BEI", defaultRange: "6M" },
        { id: "s4_btu_curva",  title: "Curva BTU completa", type: "line",
          dataSource: catalog("curva_btu_uf"), xField: "fecha",
          unit: "% real", yFormat: "percent", defaultRange: "1Y" },
        { id: "s4_spread_spc", title: "Spread BTP-SPC 5Y y 10Y", type: "line",
          dataSource: mock(403, { mockType: "multi", mockSeries: ["5Y", "10Y"] }),
          mockConfig: { start: -20, vol: 0.06 }, unit: "pb", yFormat: "bps" },
        { id: "s4_ust",        title: "Curva US Treasury", type: "line",
          dataSource: catalog("curva_ust"), xField: "fecha",
          unit: "% nominal", yFormat: "percent", defaultRange: "6M" },
    ],
};

const S5 = {
    id: "s5", title: "RF · Volúmenes", group: "Mercado",
    subtitle: "Volatilidad y montos transados",
    charts: [
        { id: "s5_vol_btptu", title: "Volatilidad diaria BTP/BTU", type: "line",
          dataSource: mock(501, { mockType: "multi", mockSeries: ["BTP", "BTU"] }),
          mockConfig: { start: 5, vol: 0.15 }, unit: "%" },
        { id: "s5_vol_bb",    title: "Volatilidad Tasas BB", type: "line",
          dataSource: mock(502), mockConfig: { start: 4, vol: 0.18 }, unit: "%" },
        { id: "s5_monto_btptu", title: "Monto transado BTP/BTU", type: "bar",
          dataSource: mock(503, { mockType: "multi", mockSeries: ["BTP", "BTU"] }),
          mockConfig: { start: 200, vol: 0.25 }, unit: "MM USD" },
        { id: "s5_monto_bb",  title: "Monto transado BB", type: "bar",
          dataSource: mock(504), mockConfig: { start: 50, vol: 0.30 }, unit: "MM USD" },
        { id: "s5_monto_bc",  title: "Monto transado BC", type: "bar",
          dataSource: mock(505), mockConfig: { start: 80, vol: 0.28 }, unit: "MM USD" },
        { id: "s5_spread_bb", title: "Spread Bancarios vs SPC", type: "line",
          dataSource: mock(506), mockConfig: { start: 25, vol: 0.05 }, unit: "pb", yFormat: "bps" },
        { id: "s5_spread_bb2",title: "Spread Bancarios vs Bonos", type: "line",
          dataSource: mock(507), mockConfig: { start: 18, vol: 0.05 }, unit: "pb", yFormat: "bps" },
        { id: "s5_spread_co", title: "Spread Corporativos vs SPC", type: "line",
          dataSource: mock(508), mockConfig: { start: 45, vol: 0.04 }, unit: "pb", yFormat: "bps" },
    ],
};

const S6 = {
    id: "s6", title: "Mercado FX", group: "Mercado",
    subtitle: "Tipo de cambio, forwards, monedas LATAM",
    charts: [
        { id: "s6_clp",       title: "USD/CLP histórico", type: "line",
          dataSource: catalog("usdclp_historico"), xField: "fecha", yFields: ["usdclp"],
          unit: "CLP/USD", defaultRange: "1Y" },
        { id: "s6_cobre_dxy", title: "Cobre vs DXY", type: "line",
          dataSource: mock(601, { mockType: "multi", mockSeries: ["Cobre", "DXY"] }),
          mockConfig: { start: 100, vol: 0.012 }, unit: "Índice" },
        { id: "s6_var_1w",    title: "Variación 1w monedas", type: "bar",
          dataSource: mock(602), mockConfig: { start: 0, vol: 0.5, days: 30 }, unit: "%" },
        { id: "s6_var_ytd",   title: "Variación YTD monedas", type: "bar",
          dataSource: mock(603), mockConfig: { start: 0, vol: 1.0, days: 30 }, unit: "%" },
        { id: "s6_amplitud",  title: "Amplitud de Puntas (%)", type: "line",
          dataSource: catalog("microestructura_fx"), xField: "fecha",
          unit: "%", defaultRange: "3M" },
        { id: "s6_vol_fx",    title: "Volatilidad Precio Transacciones", type: "line",
          dataSource: catalog("microestructura_fx"), xField: "fecha",
          unit: "%", defaultRange: "3M" },
        { id: "s6_fwd",       title: "Forward Points (pesos)", type: "line",
          dataSource: catalog("forwards_clp_curva"), xField: "fecha",
          unit: "pesos", defaultRange: "3M" },
        { id: "s6_latam",     title: "Índice monedas LATAM", type: "line",
          dataSource: catalog("indice_monedas_latam"), xField: "fecha",
          unit: "Índice", defaultRange: "1Y" },
    ],
};

const S7 = {
    id: "s7", title: "FX · Diferencial", group: "Mercado",
    subtitle: "Diferencial de tasas, flujos cambiarios, fixing",
    charts: [
        { id: "s7_swap_ois",   title: "Spread Swap CLP - OIS", type: "line",
          dataSource: catalog("curva_ois_sofr"), xField: "fecha",
          unit: "pb", yFormat: "bps", defaultRange: "6M" },
        { id: "s7_tpm_fed",    title: "Spread TPM IMP - FED IMP", type: "line",
          dataSource: catalog("expectativas_tpm_mipr"), xField: "fecha",
          unit: "pb", yFormat: "bps", defaultRange: "3M" },
        { id: "s7_flujo_14d",  title: "Flujo Cambiario 14d", type: "bar",
          dataSource: mock(701), mockConfig: { start: 0, vol: 50, days: 90 }, unit: "MM USD" },
        { id: "s7_flujo_7d",   title: "Flujo Cambiario 7d", type: "bar",
          dataSource: mock(702), mockConfig: { start: 0, vol: 30, days: 90 }, unit: "MM USD" },
        { id: "s7_fixing_sec", title: "Fixing banca (sector)", type: "line",
          dataSource: mock(703, { mockType: "multi", mockSeries: ["Banca", "AFP", "Otros"] }),
          mockConfig: { start: 0, vol: 8 }, unit: "MM USD" },
        { id: "s7_fixing_fec", title: "Fixing banca (por fecha)", type: "line",
          dataSource: mock(704), mockConfig: { start: 0, vol: 12 }, unit: "MM USD" },
        { id: "s7_datatec",    title: "Transado Datatec", type: "bar",
          dataSource: mock(705), mockConfig: { start: 800, vol: 0.2 }, unit: "MM USD" },
        { id: "s7_tasa_fwd",   title: "Tasa Implícita Forward", type: "line",
          dataSource: mock(706), mockConfig: { start: 5.5, vol: 0.04 }, unit: "%", yFormat: "percent" },
    ],
};

// ── PORTAFOLIOS ──────────────────────────────────────────────────────────

const S8 = {
    id: "s8", title: "AFP · Allocation", group: "Portafolios",
    subtitle: "Distribución de activos y attribution",
    charts: [
        { id: "s8_alloc",   title: "Allocation internacional vs local", type: "area",
          dataSource: mock(801, { mockType: "multi", mockSeries: ["Internacional", "Local"] }),
          mockConfig: { start: 50, vol: 0.01 }, unit: "%" },
        { id: "s8_attr",    title: "Attribution por clase de activo", type: "bar",
          dataSource: mock(802, { mockType: "multi", mockSeries: ["Renta fija", "RV", "Alternativos"] }),
          mockConfig: { start: 0, vol: 0.8, days: 30 }, unit: "pb" },
        { id: "s8_dv01",    title: "DV01 SPC por moneda", type: "line",
          dataSource: mock(803, { mockType: "multi", mockSeries: ["CLP", "UF"] }),
          mockConfig: { start: 100, vol: 0.03 }, unit: "MM USD" },
        { id: "s8_traspaso",title: "Traspaso de fondos AFP", type: "bar",
          dataSource: mock(804), mockConfig: { start: 0, vol: 50, days: 60 }, unit: "MM USD" },
    ],
};

const S9 = {
    id: "s9", title: "AFP · Derivados", group: "Portafolios",
    subtitle: "Posición spot y derivados",
    charts: [
        { id: "s9_spot",   title: "Posición spot y derivados AFP", type: "line",
          dataSource: mock(901, { mockType: "multi", mockSeries: ["Spot", "Derivados"] }),
          mockConfig: { start: 8000, vol: 0.02 }, unit: "MM USD" },
        { id: "s9_mtm",    title: "MtM Swap AFP", type: "line",
          dataSource: mock(902), mockConfig: { start: -200, vol: 0.08 }, unit: "MM USD" },
        { id: "s9_dvol",   title: "AFP Derivados — volumen", type: "bar",
          dataSource: mock(903), mockConfig: { start: 500, vol: 0.15 }, unit: "MM USD" },
        { id: "s9_dpos",   title: "AFP Derivados — posición", type: "area",
          dataSource: mock(904), mockConfig: { start: 3000, vol: 0.05 }, unit: "MM USD" },
    ],
};

const S10 = {
    id: "s10", title: "AFP · RFL", group: "Portafolios",
    subtitle: "Renta fija local y flujos cambiarios",
    charts: [
        { id: "s10_pos_rfl", title: "Posición RFL AFP", type: "area",
          dataSource: mock(1001), mockConfig: { start: 25000, vol: 0.03 }, unit: "MM USD" },
        { id: "s10_dcv",     title: "Variación Semanal DCV", type: "bar",
          dataSource: mock(1002), mockConfig: { start: 0, vol: 200, days: 52 }, unit: "MM USD" },
        { id: "s10_fcamb",   title: "Flujos cambiarios 1w", type: "bar",
          dataSource: mock(1003), mockConfig: { start: 0, vol: 150, days: 90 }, unit: "MM USD" },
        { id: "s10_stock",   title: "Stock Fondos", type: "line",
          dataSource: mock(1004), mockConfig: { start: 180000, vol: 0.015 }, unit: "MM USD" },
    ],
};

const S11 = {
    id: "s11", title: "FFMM", group: "Portafolios",
    subtitle: "Fondos mutuos — flujos y allocation",
    charts: [
        { id: "s11_flujos",   title: "Flujos semanales por fondo", type: "bar",
          dataSource: mock(1101, { mockType: "multi", mockSeries: ["MM", "DAP", "RF"] }),
          mockConfig: { start: 0, vol: 80, days: 52 }, unit: "MM USD" },
        { id: "s11_alloc",    title: "Allocation T6", type: "line",
          dataSource: mock(1102, { mockType: "multi", mockSeries: ["RF Local", "RF Ext.", "MM"] }),
          mockConfig: { start: 33, vol: 0.02 }, unit: "%" },
        { id: "s11_dcv",      title: "Composición portafolio DCV", type: "line",
          dataSource: mock(1103, { mockType: "multi", mockSeries: ["BTP", "BTU", "PDBC"] }),
          mockConfig: { start: 33, vol: 0.02 }, unit: "%" },
        { id: "s11_dap_pdbc", title: "Stock DAP y PDBC", type: "area",
          dataSource: mock(1104, { mockType: "multi", mockSeries: ["DAP", "PDBC"] }),
          mockConfig: { start: 3000, vol: 0.04 }, unit: "MM USD" },
    ],
};

const S12 = {
    id: "s12", title: "FFMM · Cont.", group: "Portafolios",
    subtitle: "Flujos spot, DV01 y duración por fondo",
    charts: [
        { id: "s12_flujos_spot", title: "Flujos spot acumulados", type: "line",
          dataSource: mock(1201), mockConfig: { start: 0, vol: 100 }, unit: "MM USD" },
        { id: "s12_dv01",        title: "DV01 por fondo", type: "bar",
          dataSource: mock(1202, { mockType: "multi", mockSeries: ["A", "B", "C", "D"] }),
          mockConfig: { start: 50, vol: 0.05, days: 30 }, unit: "MM USD" },
        { id: "s12_duracion",    title: "Duración por tipo de fondo", type: "line",
          dataSource: mock(1203, { mockType: "multi", mockSeries: ["Conservador", "Moderado", "Agresivo"] }),
          mockConfig: { start: 3, vol: 0.02 }, unit: "Años" },
        { id: "s12_pendiente",   title: "FFMM — pendiente", type: "line",
          dataSource: mock(1204), mockConfig: { start: 100, vol: 0.03 }, unit: "MM USD" },
    ],
};

const S13 = {
    id: "s13", title: "No Residentes", group: "Portafolios",
    subtitle: "Inversionistas internacionales",
    charts: [
        { id: "s13_pos",       title: "Posición cambiaria histórica NR", type: "line",
          dataSource: mock(1301, { mockType: "multi", mockSeries: ["Spot", "Forward"] }),
          mockConfig: { start: 8000, vol: 0.03 }, unit: "MM USD" },
        { id: "s13_spc",       title: "Posición NR en SPC nominal", type: "area",
          dataSource: mock(1302), mockConfig: { start: 4500, vol: 0.04 }, unit: "MM USD" },
        { id: "s13_fspot",     title: "NR: Flujos Spot", type: "bar",
          dataSource: mock(1303), mockConfig: { start: 0, vol: 100, days: 90 }, unit: "MM USD" },
        { id: "s13_rfl",       title: "Posición RFL No Residente", type: "line",
          dataSource: mock(1304), mockConfig: { start: 3500, vol: 0.025 }, unit: "MM USD" },
    ],
};

const S14 = {
    id: "s14", title: "NR · Cont.", group: "Portafolios",
    subtitle: "No residentes — continuación",
    charts: [
        { id: "s14_btp_dcv", title: "Var. Stock BTP en DCV", type: "bar",
          dataSource: mock(1401), mockConfig: { start: 0, vol: 80, days: 60 }, unit: "MM USD" },
        { id: "s14_p1",      title: "NR — pendiente 1", type: "line",
          dataSource: mock(1402), mockConfig: { start: 200, vol: 0.04 }, unit: "MM USD" },
        { id: "s14_p2",      title: "NR — pendiente 2", type: "line",
          dataSource: mock(1403), mockConfig: { start: 150, vol: 0.04 }, unit: "MM USD" },
        { id: "s14_p3",      title: "NR — pendiente 3", type: "line",
          dataSource: mock(1404), mockConfig: { start: 100, vol: 0.05 }, unit: "MM USD" },
    ],
};

const S15 = {
    id: "s15", title: "Bancos", group: "Portafolios",
    subtitle: "Emisiones bancarias",
    charts: [
        { id: "s15_stock",   title: "Stock Bonos Bancarios por Banco", type: "bar",
          dataSource: mock(1501, { mockType: "multi", mockSeries: ["BCI", "BCH", "Santander", "Estado"] }),
          mockConfig: { start: 800, vol: 0.03, days: 30 }, unit: "MM USD" },
        { id: "s15_dap",     title: "DAP por Banco", type: "bar",
          dataSource: mock(1502, { mockType: "multi", mockSeries: ["BCI", "BCH", "Santander", "Estado"] }),
          mockConfig: { start: 1500, vol: 0.04, days: 30 }, unit: "MM USD" },
        { id: "s15_vtos",    title: "Vencimientos corto plazo", type: "bar",
          dataSource: mock(1503), mockConfig: { start: 500, vol: 0.15, days: 60 }, unit: "MM USD" },
        { id: "s15_emis",    title: "Emisiones y vencimientos afuera", type: "line",
          dataSource: mock(1504, { mockType: "multi", mockSeries: ["Emisiones", "Vtos"] }),
          mockConfig: { start: 800, vol: 0.06 }, unit: "MM USD" },
    ],
};

const S16 = {
    id: "s16", title: "Compañías de Seguros", group: "Portafolios",
    subtitle: "CSV — stocks, flujos, duración",
    charts: [
        { id: "s16_dcv_plazo", title: "Stock DCV (por plazo)", type: "bar",
          dataSource: mock(1601, { mockType: "multi", mockSeries: ["<2Y", "2-5Y", "5-10Y", ">10Y"] }),
          mockConfig: { start: 2000, vol: 0.02, days: 30 }, unit: "MM USD" },
        { id: "s16_dcv_inst",  title: "Stock DCV (por instrumento)", type: "bar",
          dataSource: mock(1602, { mockType: "multi", mockSeries: ["BTP", "BTU", "Corp", "Banc"] }),
          mockConfig: { start: 1500, vol: 0.025, days: 30 }, unit: "MM USD" },
        { id: "s16_camb",      title: "Posición cambiaria histórica CSV", type: "line",
          dataSource: mock(1603), mockConfig: { start: 5000, vol: 0.02 }, unit: "MM USD" },
        { id: "s16_fspot",     title: "CSV: Flujos Spot", type: "bar",
          dataSource: mock(1604), mockConfig: { start: 0, vol: 80, days: 90 }, unit: "MM USD" },
        { id: "s16_duracion",  title: "Duración cartera RFL CSV", type: "line",
          dataSource: mock(1605), mockConfig: { start: 8.5, vol: 0.015 }, unit: "Años" },
        { id: "s16_flujos",    title: "Flujos DCV", type: "bar",
          dataSource: mock(1606), mockConfig: { start: 0, vol: 60, days: 60 }, unit: "MM USD" },
    ],
};

// ── Export ───────────────────────────────────────────────────────────────

BCCh.SECTIONS = [
    AGENTE_GOEM,
    OVERVIEW,
    MERCADOS,
    MERCADOS_ONLINE,
    S1, S2, S3, S4, S5, S6, S7,
    S8, S9, S10, S11, S12, S13, S14, S15, S16,
];

BCCh.SECTION_GROUPS = [
    { id: "inicio",      label: "Inicio",      ids: ["agente_goem", "overview", "mercados", "mercados_online"] },
    { id: "mercado",     label: "Mercado",     ids: ["s1","s2","s3","s4","s5","s6","s7"] },
    { id: "portafolios", label: "Portafolios", ids: ["s8","s9","s10","s11","s12","s13","s14","s15","s16"] },
];

BCCh.getSection = (id) => BCCh.SECTIONS.find((s) => s.id === id);
