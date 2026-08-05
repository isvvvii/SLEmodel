suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(patchwork)
  library(grid)
})

# CellPhoneDB and B-cell glycosylation plots.

BASE_DIR <- Sys.getenv(
  "SLEMODEL_SCRNA_OUTPUT_DIR",
  file.path(getwd(), "outputs", "single_cell")
)
GLYCO_DIR <- Sys.getenv(
  "SLEMODEL_SCRNA_GLYCO_SOURCE_DIR",
  BASE_DIR
)

SOURCE_DIR <- Sys.getenv(
  "SLEMODEL_COMMUNICATION_SOURCE_DIR",
  file.path(BASE_DIR, "source_tables")
)
OUT_DIR <- file.path(BASE_DIR, "figures", "communication_glycosylation")
SCRIPT_PATH <- "analyses/single_cell/figures/plot_communication_glycosylation.R"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

OUT_STEM <- "communication_glycosylation"
OUT_PNG <- file.path(OUT_DIR, paste0(OUT_STEM, ".png"))
OUT_PDF <- file.path(OUT_DIR, paste0(OUT_STEM, ".pdf"))
OUT_SVG <- file.path(OUT_DIR, paste0(OUT_STEM, ".svg"))
OUT_SOURCE <- file.path(OUT_DIR, paste0(OUT_STEM, "_source_used.tsv"))
OUT_README <- file.path(OUT_DIR, paste0("README_", OUT_STEM, ".txt"))
OUT_CYTOTOXIC_CANDIDATES <- file.path(OUT_DIR, "cytotoxic_candidate_interactions.tsv")
OUT_BCELL_CANDIDATES <- file.path(OUT_DIR, "bcell_support_candidate_interactions.tsv")

COL_BLUE <- "#2F6FA3"
COL_RED <- "#B94A3D"

read_table <- function(path) {
  if (!file.exists(path)) stop("Missing required input: ", path)
  suppressWarnings(readr::read_tsv(path, show_col_types = FALSE, progress = FALSE, na = c("", "NA", "NaN")))
}

num <- function(x) suppressWarnings(as.numeric(x))

fmt_num <- function(x, digits = 4) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

read_png_dim <- function(path) {
  x <- readBin(path, what = "raw", n = 24)
  if (length(x) < 24 || !all(as.integer(x[1:8]) == c(137, 80, 78, 71, 13, 10, 26, 10))) {
    return(c(width = NA_integer_, height = NA_integer_))
  }
  width <- readBin(x[17:20], what = "integer", n = 1, size = 4, endian = "big")
  height <- readBin(x[21:24], what = "integer", n = 1, size = 4, endian = "big")
  c(width = width, height = height)
}

clean_text <- function(x) {
  x %>%
    as.character() %>%
    str_replace_all("\n", " ") %>%
    str_squish()
}

short_axis <- function(x) {
  recode(
    x,
    "Platelet chemokine axis" = "Platelet\nchemokine",
    "CXCL8-CXCR2 axis" = "CXCL8-\nCXCR2",
    "CCL3/CCR1 axis" = "CCL3-\nCCR1",
    "MIF-CD74 axis" = "MIF-\nCD74/CXCR4",
    "CD48-CD244 axis" = "CD48-\nCD244",
    "CCL5-CCR5 axis" = "CCL5-\nCCR5",
    "CCL4-CCR5 axis" = "CCL4-\nCCR5",
    "TNF-TNFR axis" = "TNF-\nTNFR",
    "CD70-CD27 axis" = "CD70-\nCD27",
    "BAFF/APRIL axis" = "BAFF/\nAPRIL",
    "CD40 axis" = "CD40-\nCD40LG",
    "Plasmablast-support axis" = "PB\nsupport",
    .default = str_replace_all(x, " axis$", "")
  )
}

short_cell <- function(x) {
  x %>%
    str_replace_all("Inflammatory monocytes", "Inflam. mono.") %>%
    str_replace_all("TREM1-high CD14 monocytes", "TREM1-hi mono.") %>%
    str_replace_all("CD14 classical monocytes", "CD14 mono.") %>%
    str_replace_all("FCGR3A/CD16 monocytes", "CD16 mono.") %>%
    str_replace_all("Platelet/MK-like cells", "Plt/MK") %>%
    str_replace_all("Annotated plasmablasts", "PB") %>%
    str_replace_all("Plasmablast-differentiation-high B cells", "PB-diff B") %>%
    str_replace_all("ISGhigh naive CD8 like", "ISG-hi naive CD8") %>%
    str_replace_all("Cytotoxic CD8 T cells", "Cyto CD8") %>%
    str_replace_all("Cytotoxic NK cells", "Cyto NK") %>%
    str_replace_all("Naive CD8 T cells", "Naive CD8") %>%
    str_replace_all("GZMKpos GZMBlow effmem like CD8", "GZMK+ effmem CD8") %>%
    str_replace_all("atypical B ABC like", "ABC-like B") %>%
    str_replace_all("naive B", "Naive B")
}

short_pair_label <- function(x) {
  parts <- str_split_fixed(x, fixed(" → "), 2)
  left <- short_cell(parts[, 1])
  right <- short_cell(parts[, 2])
  paste0(left, "\n-> ", right)
}

cell_display <- function(x) {
  recode(
    as.character(x),
    "NK_Cytotoxic" = "Cyto NK",
    "cytotoxic_CD8" = "Cyto CD8",
    "ISGhigh_naive_CD8_like" = "ISG-hi naive CD8",
    "FCGR3A_CD16_monocyte" = "CD16 mono.",
    "inflammatory_monocyte" = "Inflam. mono.",
    "TREM1high_like_CD14_monocyte" = "TREM1-hi mono.",
    "CD14_classical_monocyte" = "CD14 mono.",
    "CD4_total" = "CD4 total",
    "CD4_T" = "CD4 T",
    "naive_B" = "Naive B",
    "memory_B" = "Memory B",
    "atypical_B_ABC_like" = "ABC-like B",
    "plasmablast_differentiation_high_B" = "PB-diff B",
    "true_annotation_plasmablast_B" = "PB",
    "cDC" = "cDCs",
    "pDC" = "pDCs",
    .default = str_replace_all(as.character(x), "_", " ")
  )
}

short_lr <- function(x) {
  x %>%
    str_replace_all("_", "-") %>%
    str_replace_all("IL1-receptor-inhibitor", "IL1R inhib.") %>%
    str_replace_all("Desoxycorticosterone-byCYP21A2-NR3C2", "CYP21A2-NR3C2")
}

prepare_cpdb_panel <- function(path, panel_id, panel_title, source_analysis, direction_label, top_per_axis = 1) {
  raw <- read_table(path) %>%
    mutate(
      APO_mean = num(APO_mean),
      nonAPO_mean = num(nonAPO_mean),
      APO_minus_nonAPO = num(APO_minus_nonAPO),
      abs_delta = num(abs_delta)
    )
  axis_levels <- unique(raw$axis)
  distinct_rows <- raw %>%
    distinct(axis, interacting_pair, sender_receiver_label, APO_mean, nonAPO_mean, APO_minus_nonAPO, abs_delta, .keep_all = TRUE)

  selected <- distinct_rows %>%
    group_by(axis) %>%
    arrange(desc(abs_delta), interacting_pair, sender_receiver_label, .by_group = TRUE) %>%
    slice_head(n = top_per_axis) %>%
    ungroup() %>%
    mutate(
      panel_id = panel_id,
      panel_title = panel_title,
      source_analysis = source_analysis,
      source_table = path,
      formal_direction = direction_label,
      formal_source_rows = nrow(raw),
      formal_distinct_rows = nrow(distinct_rows),
      selection_rule = paste0("Top ", top_per_axis, " distinct source-table row per formal axis by |APO - non-APO|"),
      axis = factor(axis, levels = axis_levels),
      axis_label = short_lr(interacting_pair),
      sender_receiver_short = short_pair_label(sender_receiver_label),
      lr_label = short_lr(interacting_pair)
    )

  x_levels <- selected %>%
    group_by(sender_receiver_short) %>%
    summarise(max_abs_delta = max(abs_delta, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(max_abs_delta), sender_receiver_short) %>%
    pull(sender_receiver_short)

  selected %>%
    mutate(
      sender_receiver_short = factor(sender_receiver_short, levels = x_levels),
      axis_label = factor(axis_label, levels = rev(unique(axis_label)))
    )
}

finalize_cpdb_selection <- function(dat, panel_id, panel_title, source_analysis, source_table, formal_direction, selection_rule) {
  dat %>%
    mutate(
      APO_mean = num(APO_mean),
      nonAPO_mean = num(nonAPO_mean),
      APO_minus_nonAPO = num(APO_minus_nonAPO),
      abs_delta = abs(APO_minus_nonAPO),
      sender_label = if_else(is.na(sender_label) | sender_label == "", cell_display(sender), sender_label),
      receiver_label = if_else(is.na(receiver_label) | receiver_label == "", cell_display(receiver), receiver_label),
      sender_receiver_label = if_else(
        is.na(sender_receiver_label) | sender_receiver_label == "",
        paste0(sender_label, " → ", receiver_label),
        sender_receiver_label
      ),
      panel_id = panel_id,
      panel_title = panel_title,
      source_analysis = source_analysis,
      source_table = source_table,
      formal_direction = formal_direction,
      formal_source_rows = nrow(dat),
      formal_distinct_rows = nrow(distinct(dat, interacting_pair, sender, receiver, .keep_all = TRUE)),
      selection_rule = selection_rule,
      axis_label = short_lr(interacting_pair),
      sender_receiver_short = short_pair_label(sender_receiver_label),
      lr_label = short_lr(interacting_pair)
    ) %>%
    distinct(interacting_pair, sender_receiver_label, .keep_all = TRUE) %>%
    mutate(
      sender_receiver_short = factor(sender_receiver_short, levels = unique(sender_receiver_short)),
      axis_label = factor(axis_label, levels = rev(unique(axis_label)))
    )
}

pick_row <- function(dat, pair, sender = NULL, receiver = NULL) {
  x <- dat %>% filter(interacting_pair == pair)
  if (!is.null(sender)) x <- x %>% filter(sender == !!sender)
  if (!is.null(receiver)) x <- x %>% filter(receiver == !!receiver)
  x %>% arrange(desc(abs(num(APO_minus_nonAPO)))) %>% slice_head(n = 1)
}

prepare_cytotoxic <- function(cytotoxic_input, key_path) {
  formal <- read_table(cytotoxic_input)
  key <- read_table(key_path)
  selected <- bind_rows(
    pick_row(key, "CCL5_CCR5", "NK_Cytotoxic", "cDC") %>% mutate(axis = "CCL5-CCR5 axis"),
    pick_row(key, "CCL4_CCR5", "NK_Cytotoxic", "cDC") %>% mutate(axis = "CCL4-CCR5 axis"),
    pick_row(formal, "CD48_CD244", "inflammatory_monocyte", "NK_Cytotoxic") %>% mutate(axis = "CD48-CD244 axis"),
    pick_row(formal, "LTA_TNFRSF1B", "true_annotation_plasmablast_B", "FCGR3A_CD16_monocyte") %>% mutate(axis = "TNF-TNFR axis")
  )
  finalize_cpdb_selection(
    selected, "adaptive_cytotoxic", "Cytotoxic/adaptive crosstalk",
    "cellphonedb_adaptive_cytotoxic and CellPhoneDB_key_axis_summary",
    paste(cytotoxic_input, key_path, sep = "; "),
    "non-APO-high adaptive/cytotoxic chemokine communication",
    "CCL5-CCR5 and CCL4-CCR5 rows from the key-axis table; CD48-CD244 and TNF/TNFR rows from the adaptive/cytotoxic source table"
  )
}

prepare_bcell <- function(bcell_input, adaptive_b_path, key_path) {
  formal <- read_table(bcell_input)
  key <- read_table(key_path)
  selected <- bind_rows(
    pick_row(formal, "TNFSF13B_TNFRSF17", "inflammatory_monocyte", "cDC") %>% mutate(axis = "BAFF/APRIL axis"),
    pick_row(formal, "TNFSF13B_TNFRSF13B", "inflammatory_monocyte", "cDC") %>% mutate(axis = "BAFF/APRIL axis"),
    pick_row(key, "CXCL12_CXCR4", "CD4_total", "naive_B") %>% mutate(axis = "CXCL12-CXCR4 axis"),
    pick_row(key, "CXCL13_CXCR5", "TREM1high_like_CD14_monocyte", "naive_B") %>% mutate(axis = "CXCL13-CXCR5 axis"),
    pick_row(formal, "CD40LG_CD40", "CD4_total", "plasmablast_differentiation_high_B") %>% mutate(axis = "CD40 axis")
  )
  finalize_cpdb_selection(
    selected, "bcell_support", "B/plasmablast survival signals",
    "cellphonedb_bcell_support, cellphonedb_adaptive_b_support and CellPhoneDB_key_axis_summary",
    paste(bcell_input, adaptive_b_path, key_path, sep = "; "),
    "non-APO-high B/plasmablast support and homing signals",
    "BAFF/APRIL receptor pairs from the B-cell support table, CXCL12-CXCR4 and CXCL13-CXCR5 from the key-axis table, and CD40LG-CD40"
  )
}

myeloid_path <- file.path(SOURCE_DIR, "cellphonedb_myeloid_platelet_source.tsv")
cytotoxic_path <- file.path(SOURCE_DIR, "cellphonedb_adaptive_cytotoxic_source.tsv")
bcell_path <- file.path(SOURCE_DIR, "cellphonedb_bcell_support_source.tsv")
adaptive_b_path <- file.path(SOURCE_DIR, "cellphonedb_adaptive_b_support_source.tsv")
key_axis_path <- file.path(BASE_DIR, "tables", "CellPhoneDB_key_axis_summary.tsv")
glyco_path <- Sys.getenv(
  "SLEMODEL_GLYCO_SCORE_SOURCE",
  file.path(GLYCO_DIR, "tables", "Bcell_glyco_projection_source.tsv")
)

write_candidate_tables <- function() {
  py <- tempfile(fileext = ".py")
  code <- c(
    "import csv, re, os, math",
    paste0("exp = ", shQuote(BASE_DIR)),
    paste0("out_cytotoxic = ", shQuote(OUT_CYTOTOXIC_CANDIDATES)),
    paste0("out_bcell = ", shQuote(OUT_BCELL_CANDIDATES)),
    "def fnum(x):",
    "    try: return float(x)",
    "    except Exception: return float('nan')",
    "def emit_candidates(files, pattern, out_path, mode):",
    "    rows=[]",
    "    for f in files:",
    "        path=os.path.join(exp, f)",
    "        if not os.path.exists(path): continue",
    "        with open(path, newline='') as fh:",
    "            rdr=csv.DictReader(fh, delimiter='\\t')",
    "            for r in rdr:",
    "                text=' '.join(str(r.get(k,'')) for k in ['interacting_pair','gene_a','gene_b','partner_a','partner_b','sender','receiver','axis','sender_label','receiver_label','sender_receiver_label']).upper()",
    "                if not pattern.search(text): continue",
    "                ga=str(r.get('gene_a','')).upper(); gb=str(r.get('gene_b','')).upper(); ip=str(r.get('interacting_pair','')).upper()",
    "                d=fnum(r.get('APO_minus_nonAPO',''))",
    "                r['source_file']=f",
    "                r['axis']=r.get('axis','')",
    "                r['abs_delta']=str(abs(d)) if d==d else ''",
    "                if mode=='cytotoxic':",
    "                    true_ccr5 = (gb=='CCR5') or ('_CCR5' in ip)",
    "                    classic = bool(re.search(r'CCL5|CCL4|CCR5|CD48|CD244|TNF|TNFR|LTA', text))",
    "                    if ga=='CCL5' and true_ccr5: notes='true CCL5-CCR5 pair'",
    "                    elif ga=='CCL4' and true_ccr5: notes='true CCL4-CCR5 pair'",
    "                    elif ga=='CCL5': notes='CCL5 chemokine row, receptor is not CCR5'",
    "                    elif ga=='CCL4': notes='CCL4 chemokine row, receptor is not CCR5'",
    "                    elif 'CD48' in ip or 'CD244' in ip: notes='CD48-CD244 cytotoxic coordination row'",
    "                    elif re.search(r'TNF|TNFR|LTA', text): notes='TNF/TNFR-family row'",
    "                    else: notes='keyword candidate'",
    "                    r['is_true_CCR5_pair']=str(true_ccr5)",
    "                    r['is_classic_cytotoxic_axis']=str(classic)",
    "                    r['notes']=notes",
    "                else:",
    "                    nonapo = d==d and d < 0",
    "                    true_support = bool(re.search(r'TNFSF13B|TNFSF13|TNFRSF13B|TNFRSF13C|TNFRSF17|CD40|CD40LG|ICOS|ICOSLG|CXCL12|CXCR4|CXCL13|CXCR5|IL6|IL6R|IL21|IL21R|PLASMABLAST|B CELL', text))",
    "                    if 'ICOSLG_ICOS' in ip: notes='true ICOSLG-ICOS exists but weak in this dataset'",
    "                    elif 'CXCL12_CXCR4' in ip: notes='true CXCL12-CXCR4 homing/support candidate'",
    "                    elif 'CXCL13_CXCR5' in ip: notes='true CXCL13-CXCR5 B-cell homing candidate'",
    "                    elif 'IL21' in ip or 'IL21R' in ip: notes='IL21/IL21R keyword candidate'",
    "                    elif 'IL6' in ip or 'IL6R' in ip: notes='IL6/IL6R keyword candidate; verify ligand identity'",
    "                    elif 'TNFSF13' in ip or 'TNFRSF13' in ip or 'TNFRSF17' in ip: notes='BAFF/APRIL receptor-family candidate'",
    "                    elif 'CD40' in ip: notes='CD40/CD40LG candidate'",
    "                    else: notes='keyword candidate'",
    "                    r['is_nonAPO_high']=str(nonapo)",
    "                    r['is_B_or_plasmablast_support']=str(true_support)",
    "                    r['notes']=notes",
    "                rows.append(r)",
    "    common=['source_file','axis','interacting_pair','gene_a','gene_b','sender','receiver','sender_label','receiver_label','APO_mean','nonAPO_mean','APO_minus_nonAPO','abs_delta','APO_significant','nonAPO_significant','status']",
    "    extra = ['is_true_CCR5_pair','is_classic_cytotoxic_axis','notes'] if mode=='cytotoxic' else ['is_nonAPO_high','is_B_or_plasmablast_support','notes']",
    "    seen=set(); out=[]",
    "    for r in rows:",
    "        key=tuple(r.get(k,'') for k in common+extra)",
    "        if key in seen: continue",
    "        seen.add(key); out.append(r)",
    "    out.sort(key=lambda r: (-(fnum(r.get('abs_delta','')) if fnum(r.get('abs_delta',''))==fnum(r.get('abs_delta','')) else -1), r.get('source_file','')))",
    "    with open(out_path,'w',newline='') as fh:",
    "        w=csv.DictWriter(fh, fieldnames=common+extra, delimiter='\\t', extrasaction='ignore')",
    "        w.writeheader(); w.writerows(out)",
    "cytotoxic_files=['source_tables/cellphonedb_adaptive_cytotoxic_source.tsv','tables/CellPhoneDB_key_axis_summary.tsv','tables/CellPhoneDB_broad_group_comparison.tsv','tables/CellPhoneDB_targeted_group_comparison.tsv']",
    "bcell_files=['source_tables/cellphonedb_bcell_support_source.tsv','source_tables/cellphonedb_adaptive_b_support_source.tsv','tables/CellPhoneDB_key_axis_summary.tsv','tables/CellPhoneDB_broad_group_comparison.tsv','tables/CellPhoneDB_targeted_group_comparison.tsv']",
    "emit_candidates(cytotoxic_files, re.compile(r'CCL5|CCL4|CCR5|GPR75|ACKR2|CD48|CD244|TNF|TNFR', re.I), out_cytotoxic, 'cytotoxic')",
    "emit_candidates(bcell_files, re.compile(r'TNFSF13B|TNFSF13|TNFRSF13B|TNFRSF13C|TNFRSF17|CD40|CD40LG|ICOS|ICOSLG|CXCL12|CXCR4|CXCL13|CXCR5|IL6|IL6R|IL21|IL21R|PLASMABLAST|PLASMA|B CELL|BAFF|APRIL', re.I), out_bcell, 'bcell')"
  )
  writeLines(code, py)
  status <- system2("python3", py)
  if (!identical(status, 0L)) stop("Candidate-table generation failed")
}
write_candidate_tables()

cpdb_myeloid <- prepare_cpdb_panel(
  myeloid_path, "myeloid_platelet", "APO inflammatory crosstalk",
  "cellphonedb_myeloid_platelet",
  "APO-high communication"
)
cpdb_cytotoxic <- prepare_cytotoxic(cytotoxic_path, key_axis_path)
cpdb_bcell <- prepare_bcell(bcell_path, adaptive_b_path, key_axis_path)
cpdb_all <- bind_rows(cpdb_myeloid, cpdb_cytotoxic, cpdb_bcell)

cpdb_lim <- max(abs(cpdb_all$APO_minus_nonAPO), na.rm = TRUE)
if (!is.finite(cpdb_lim) || cpdb_lim <= 0) cpdb_lim <- 1
cpdb_tick <- signif(cpdb_lim, 2)
cpdb_breaks <- c(-cpdb_tick, 0, cpdb_tick)
cpdb_labels <- sprintf("%.1f", cpdb_breaks)
size_range <- range(cpdb_all$abs_delta, na.rm = TRUE)
if (!all(is.finite(size_range)) || diff(size_range) == 0) size_range <- c(0, max(cpdb_all$abs_delta, na.rm = TRUE, 1))

theme_comm <- function() {
  theme_bw(base_size = 5.2) +
    theme(
      plot.title = element_text(size = 7.0, face = "plain", hjust = 0.5, margin = margin(b = 2)),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.15),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "grey45", fill = NA, linewidth = 0.28),
      axis.title = element_blank(),
      axis.text.x = element_text(size = 4.15, angle = 55, hjust = 1, vjust = 1, color = "black", lineheight = 0.84),
      axis.text.y = element_text(size = 5.1, color = "black", lineheight = 0.86),
      plot.margin = margin(2, 2, 2, 2),
      legend.title = element_text(size = 5.2),
      legend.text = element_text(size = 4.8),
      legend.key.height = unit(2.4, "mm"),
      legend.key.width = unit(4.0, "mm")
    )
}

plot_cpdb <- function(dat, title) {
  ggplot(dat, aes(sender_receiver_short, axis_label)) +
    geom_point(aes(size = abs_delta, color = APO_minus_nonAPO), alpha = 0.90) +
    scale_color_gradient2(
      name = "APO -\nnon-APO",
      low = COL_BLUE,
      mid = "white",
      high = COL_RED,
      midpoint = 0,
      limits = c(-cpdb_lim, cpdb_lim),
      breaks = cpdb_breaks,
      labels = cpdb_labels,
      guide = guide_colorbar(barwidth = unit(12, "mm"), barheight = unit(2.2, "mm"), title.position = "top", display = "rectangles")
    ) +
    scale_size_continuous(
      name = "|Delta|",
      limits = size_range,
      range = c(1.1, 4.3),
      guide = guide_legend(title.position = "top", override.aes = list(alpha = 0.9))
    ) +
    scale_x_discrete(expand = expansion(add = c(0.45, 0.45))) +
    scale_y_discrete(expand = expansion(add = c(0.35, 0.35))) +
    labs(title = title) +
    coord_cartesian(clip = "off") +
    theme_comm()
}

p_g <- plot_cpdb(cpdb_myeloid, "APO inflammatory crosstalk")
p_h <- plot_cpdb(cpdb_cytotoxic, "Cytotoxic/adaptive crosstalk")
p_i <- plot_cpdb(cpdb_bcell, "B/plasmablast survival signals")

projection_order <- c(
  "Low sialylation risk",
  "Low sialic-acid supply risk",
  "Low galactosylation risk",
  "MGAT3 bisecting"
)
subtype_order <- c("Naive B", "Memory B", "ABC-like B", "Plasmablast")
projection_labels <- c(
  "Low sialylation risk" = "Low Sia",
  "Low sialic-acid supply risk" = "Low sialic-\nacid supply",
  "Low galactosylation risk" = "Low Gal",
  "MGAT3 bisecting" = "High\nbisecting"
)

glyco_source <- read_table(glyco_path) %>%
  mutate(
    delta_z = num(delta_z),
    delta_mean = num(delta_mean),
    mean_APO = num(mean_APO),
    mean_nonAPO = num(mean_nonAPO),
    wilcoxon_p = num(wilcoxon_p),
    BH_FDR = num(BH_FDR),
    module_label = recode(module, !!!projection_labels),
    B_subtype_label = factor(B_subtype_label, levels = subtype_order),
    module = factor(module, levels = projection_order),
    module_label = factor(module_label, levels = unname(projection_labels)),
    panel_id = "bcell_glycosylation",
    panel_title = "B-cell glyco-projection",
    source_table = glyco_path,
    source_analysis = "B-cell glycosylation projection"
  ) %>%
  filter(!is.na(B_subtype_label), !is.na(module))

glyco_lim <- max(abs(glyco_source$delta_z), na.rm = TRUE)
if (!is.finite(glyco_lim) || glyco_lim <= 0) glyco_lim <- 0.1
glyco_tick <- signif(glyco_lim, 2)
glyco_breaks <- c(-glyco_tick, 0, glyco_tick)
glyco_labels <- sprintf("%.1f", glyco_breaks)

theme_heat <- function() {
  theme_bw(base_size = 5.5) +
    theme(
      plot.title = element_text(size = 7.0, face = "plain", hjust = 0.5, margin = margin(b = 2)),
      panel.grid = element_blank(),
      panel.border = element_rect(color = "grey45", fill = NA, linewidth = 0.28),
      axis.title = element_blank(),
      axis.text.x = element_text(size = 5.0, angle = 35, hjust = 1, vjust = 1, color = "black", lineheight = 0.88),
      axis.text.y = element_text(size = 5.8, color = "black"),
      plot.margin = margin(2, 2, 2, 2),
      legend.title = element_text(size = 5.2),
      legend.text = element_text(size = 4.8),
      legend.key.height = unit(2.4, "mm"),
      legend.key.width = unit(4.0, "mm")
    )
}

p_j <- ggplot(glyco_source, aes(module_label, B_subtype_label, fill = delta_z)) +
  geom_tile(color = "white", linewidth = 0.30) +
  scale_y_discrete(limits = rev(subtype_order), expand = expansion(add = c(0, 0))) +
  scale_x_discrete(expand = expansion(add = c(0, 0))) +
  scale_fill_gradient2(
    name = "APO -\nnon-APO score",
    low = COL_BLUE,
    mid = "white",
    high = COL_RED,
    midpoint = 0,
    limits = c(-glyco_lim, glyco_lim),
    breaks = glyco_breaks,
    labels = glyco_labels,
    guide = guide_colorbar(barwidth = unit(12, "mm"), barheight = unit(2.2, "mm"), title.position = "top", display = "rectangles")
  ) +
  labs(title = "B-cell glyco-projection") +
  coord_cartesian(clip = "off") +
  theme_heat()

combined <- (p_g | p_h | p_i | p_j) +
  plot_layout(widths = c(1.05, 1.0, 1.05, 0.75), guides = "collect") &
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    legend.spacing.x = unit(2.0, "mm"),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
    plot.background = element_rect(fill = "white", color = NA)
  )

width_in <- 180 / 25.4
height_in <- 80 / 25.4

ggsave(OUT_PNG, combined, width = width_in, height = height_in, units = "in", dpi = 600, limitsize = FALSE, bg = "white")

grDevices::pdf(
  OUT_PDF,
  width = width_in,
  height = height_in,
  onefile = FALSE,
  paper = "special",
  bg = "white",
  useDingbats = FALSE,
  compress = FALSE
)
print(combined)
dev.off()

svg_device <- if (requireNamespace("svglite", quietly = TRUE)) "svglite::svglite" else "grDevices::svg"
if (requireNamespace("svglite", quietly = TRUE)) {
  svglite::svglite(OUT_SVG, width = width_in, height = height_in, bg = "white")
  print(combined)
  dev.off()
} else {
  grDevices::svg(OUT_SVG, width = width_in, height = height_in, bg = "white", onefile = FALSE)
  print(combined)
  dev.off()
}

cpdb_source_out <- cpdb_all %>%
  transmute(
    panel_id,
    panel_title,
    data_type = "CellPhoneDB_selected_interactions",
    source_analysis,
    source_table,
    formal_direction,
    selection_rule,
    formal_source_rows,
    formal_distinct_rows,
    axis = as.character(axis),
    axis_label = clean_text(axis_label),
    interacting_pair,
    lr_label = clean_text(lr_label),
    gene_a = as.character(gene_a),
    gene_b = as.character(gene_b),
    sender_receiver_label,
    sender_receiver_short = clean_text(sender_receiver_short),
    APO_mean,
    nonAPO_mean,
    APO_minus_nonAPO,
    abs_delta,
    B_subtype_label = NA_character_,
    module = NA_character_,
    module_label = NA_character_,
    glyco_value_delta_z = NA_real_,
    glyco_value_delta_mean = NA_real_,
    wilcoxon_p = NA_real_,
    BH_FDR = NA_real_
  )

glyco_source_out <- glyco_source %>%
  transmute(
    panel_id,
    panel_title,
    data_type = "Bcell_glycosylation_projection",
    source_analysis,
    source_table,
    formal_direction = "APO minus non-APO direction-coded z-score",
    selection_rule = "All displayed B-cell subsets and glycosylation modules",
    formal_source_rows = nrow(glyco_source),
    formal_distinct_rows = nrow(glyco_source),
    axis = NA_character_,
    axis_label = NA_character_,
    interacting_pair = NA_character_,
    lr_label = NA_character_,
    gene_a = NA_character_,
    gene_b = NA_character_,
    sender_receiver_label = NA_character_,
    sender_receiver_short = NA_character_,
    APO_mean = mean_APO,
    nonAPO_mean = mean_nonAPO,
    APO_minus_nonAPO = delta_z,
    abs_delta = abs(delta_z),
    B_subtype_label = as.character(B_subtype_label),
    module = as.character(module),
    module_label = clean_text(module_label),
    glyco_value_delta_z = delta_z,
    glyco_value_delta_mean = delta_mean,
    wilcoxon_p,
    BH_FDR
  )

source_used <- bind_rows(cpdb_source_out, glyco_source_out)
readr::write_tsv(source_used, OUT_SOURCE, na = "")

png_dim <- read_png_dim(OUT_PNG)

axis_lines <- cpdb_source_out %>%
  arrange(panel_id, factor(axis, levels = unique(axis)), desc(abs_delta)) %>%
  transmute(line = paste0(
    "- ", panel_title, ": ", axis, " | ", interacting_pair,
    " | ", sender_receiver_label,
    " | APO-minus-nonAPO=", fmt_num(APO_minus_nonAPO, 4),
    " | |Delta|=", fmt_num(abs_delta, 4)
  )) %>%
  pull(line)

glyco_lines <- glyco_source_out %>%
  arrange(factor(B_subtype_label, levels = subtype_order), factor(module, levels = projection_order)) %>%
  transmute(line = paste0(
    "- ", B_subtype_label, " x ", module,
    " | delta_z=", fmt_num(glyco_value_delta_z, 4),
    " | delta_mean=", fmt_num(glyco_value_delta_mean, 4),
    " | p=", fmt_num(wilcoxon_p, 4),
    " | BH_FDR=", fmt_num(BH_FDR, 4)
  )) %>%
  pull(line)

readme <- c(
  "CellPhoneDB and B-cell glycosylation plots",
  paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "",
  "Script:",
  SCRIPT_PATH,
  "",
  "Input tables:",
  paste0("- ", myeloid_path),
  paste0("- ", cytotoxic_path),
  paste0("- ", bcell_path),
  paste0("- ", adaptive_b_path),
  paste0("- ", key_axis_path),
  paste0("- ", glyco_path),
  "",
  "Candidate-interaction tables:",
  paste0("- Cytotoxic candidates: ", OUT_CYTOTOXIC_CANDIDATES),
  paste0("- B-cell-support candidates: ", OUT_BCELL_CANDIDATES),
  "",
  "CellPhoneDB analyses:",
  "- The myeloid/platelet analysis contains one representative APO-high interaction from each prespecified inflammatory axis.",
  "- The adaptive/cytotoxic analysis contains CCL5-CCR5, CCL4-CCR5, CD48-CD244 and LTA-TNFRSF1B interactions.",
  "- The B-cell-support analysis contains BAFF/APRIL, CXCL12-CXCR4, CXCL13-CXCR5 and CD40LG-CD40 interactions related to B-cell support or homing.",
  "- Bubble size represents |APO mean - non-APO mean|.",
  "- Bubble color represents APO mean - non-APO mean. Red means APO-high/enriched, blue means non-APO-high/enriched.",
  "- These CellPhoneDB values are descriptive interaction differences; no formal between-group p value is shown.",
  "",
  "Selected source-table interactions:",
  axis_lines,
  "",
  "B-cell glycosylation analysis:",
  "- Heatmap value is delta_z = APO mean z-score - non-APO mean z-score for risk-direction corrected model glyco projection features.",
  "- This is a glycosylation-related transcriptional context/projection, not direct IgG glycan abundance.",
  "- Rows: Naive B, Memory B, ABC-like B, Plasmablast. Total B cells was removed for summary-plot compactness.",
  "- Columns: Low Sia, Low sialic-acid supply, Low Gal, High bisecting (MGAT3 bisecting).",
  "- Corrected glyco-risk projection, corrected extended glyco-risk projection, and MGAT3-to-branching balance were omitted to retain the four core glyco-risk directions.",
  paste0("- Color scale is symmetric around 0 with limits +/-", fmt_num(glyco_lim, 4), ". Blue = APO lower; white = no difference; red = APO higher."),
  "",
  "B-cell glycosylation heatmap values:",
  glyco_lines,
  "",
  "Outputs:",
  paste0("- PNG: ", OUT_PNG),
  paste0("- PDF: ", OUT_PDF),
  paste0("- SVG: ", OUT_SVG),
  paste0("- Source used TSV: ", OUT_SOURCE),
  paste0("- Cytotoxic candidate TSV: ", OUT_CYTOTOXIC_CANDIDATES),
  paste0("- B-cell-support candidate TSV: ", OUT_BCELL_CANDIDATES),
  paste0("- README: ", OUT_README),
  paste0("- PNG dimensions: ", png_dim[["width"]], " x ", png_dim[["height"]], " px"),
  paste0("- SVG device: ", svg_device),
  "- PDF was generated directly with grDevices::pdf(compress=FALSE, useDingbats=FALSE), with no raster conversion.",
  "- SVG was generated directly from the ggplot/patchwork object.",
  "",
  "Layout check notes:",
  "- The layout contains one row and four columns, without panel letters or a global title.",
  "- Display labels are shortened for readability; full source labels are retained in the source-used TSV and above."
)
writeLines(readme, OUT_README)

message("Generated files:")
message(OUT_PNG)
message(OUT_PDF)
message(OUT_SVG)
message(OUT_SOURCE)
message(OUT_README)
message("PNG dimensions: ", png_dim[["width"]], " x ", png_dim[["height"]])
message("SVG device: ", svg_device)
