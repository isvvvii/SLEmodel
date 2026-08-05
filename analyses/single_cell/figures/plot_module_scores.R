suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(grid)
})

BASE_DIR <- Sys.getenv(
  "SLEMODEL_SCRNA_OUTPUT_DIR",
  file.path(getwd(), "outputs", "single_cell")
)

MODULE_SOURCE <- Sys.getenv(
  "SLEMODEL_MODULE_SCORE_SOURCE",
  file.path(BASE_DIR, "tables", "module_scores_source.tsv")
)
REFINEMENT_DIR <- file.path(BASE_DIR, "bcell_module_refinement")
REFINED_SCORES <- file.path(REFINEMENT_DIR, "targeted_Blineage_module_scores_by_sample.tsv")
REFINED_STATS <- file.path(REFINEMENT_DIR, "targeted_Blineage_module_stats_all.tsv")

OUT_DIR <- file.path(BASE_DIR, "figures", "module_scores")
SCRIPT_PATH <- "analyses/single_cell/figures/plot_module_scores.R"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

OUT_STEM <- "module_scores"
OUT_PNG <- file.path(OUT_DIR, paste0(OUT_STEM, ".png"))
OUT_PDF <- file.path(OUT_DIR, paste0(OUT_STEM, ".pdf"))
OUT_SVG <- file.path(OUT_DIR, paste0(OUT_STEM, ".svg"))
OUT_SOURCE <- file.path(OUT_DIR, paste0(OUT_STEM, "_source_selected.tsv"))
OUT_STATS <- file.path(OUT_DIR, paste0(OUT_STEM, "_stats_summary.tsv"))
OUT_README <- file.path(OUT_DIR, "README_module_scores.txt")

COL_NONAPO <- "#5F7F9A"
COL_APO <- "#C9684D"

read_table <- function(path) {
  if (!file.exists(path)) stop("Missing required input: ", path)
  suppressWarnings(readr::read_tsv(path, show_col_types = FALSE, progress = FALSE, na = c("", "NA", "NaN")))
}

num <- function(x) suppressWarnings(as.numeric(x))

fmt_num <- function(x, digits = 4) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

format_p <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "p = NA",
    p < 0.001 ~ "p < 0.001",
    p < 0.1 ~ paste0("p = ", formatC(p, format = "f", digits = 3)),
    TRUE ~ paste0("p = ", formatC(p, format = "f", digits = 2))
  )
}

star_label <- function(p) {
  dplyr::case_when(
    is.na(p) ~ "ns",
    p < 0.001 ~ "***",
    p < 0.01 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ "ns"
  )
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

theme_dotbox <- function() {
  theme_bw(base_size = 6.7) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "grey88", linewidth = 0.20),
      panel.border = element_rect(color = "grey35", fill = NA, linewidth = 0.30),
      strip.background = element_blank(),
      strip.text = element_text(size = 6.15, face = "plain", color = "black", lineheight = 0.86, margin = margin(b = 1.2)),
      axis.title.x = element_blank(),
      axis.title.y = element_text(size = 6.9, color = "black", margin = margin(r = 3)),
      axis.text.x = element_text(size = 6.25, color = "black"),
      axis.text.y = element_text(size = 6.05, color = "black"),
      plot.title = element_blank(),
      plot.margin = margin(1, 2, 2, 2),
      legend.position = "none"
    )
}

full_titles <- tribble(
  ~priority, ~panel_label,
  1, "Inflammatory monocytes\nInflammatory response",
  2, "Inflammatory monocytes\nNeutrophil degranulation",
  3, "Inflammatory monocytes\nTREM1 inflammatory",
  4, "NK/TNK cells\nCX3CR1 migration",
  5, "NK/TNK cells\nTNF/TNFR axis",
  6, "NK/TNK cells\nEffector memory",
  7, "Naive CD8 T cells\nT-cell activation",
  8, "Plasmablast B cells\nBAFF/APRIL receptor response",
  9, "FCGR3A+ monocytes\nBAFF/APRIL ligand source",
  10, "CD14+ monocytes\nBAFF/APRIL ligand source"
)

f_specs <- tribble(
  ~priority, ~category, ~cell_scope, ~module_name, ~expected_direction, ~panel_label,
  8, "B-cell/plasmablast-support modules decreased", "B|true_annotation_plasmablast_B", "BAFF_APRIL_receptor_response_B", "APO_low", full_titles$panel_label[8],
  9, "B-cell/plasmablast-support modules decreased", "myeloid|FCGR3A_CD16_monocyte", "BAFF_APRIL_ligand_source", "APO_low", full_titles$panel_label[9],
  10, "B-cell/plasmablast-support modules decreased", "myeloid|CD14_classical_monocyte", "BAFF_APRIL_ligand_source", "APO_low", full_titles$panel_label[10]
)

module_source <- read_table(MODULE_SOURCE)
module_source_cols <- names(module_source)

base_modules <- module_source %>%
  mutate(priority = as.integer(priority)) %>%
  filter(priority <= 7)

refined_scores <- read_table(REFINED_SCORES) %>%
  rename(group_label = group) %>%
  mutate(
    module_score_mean = num(module_score_mean),
    module_score_median = num(module_score_median),
    n_cells = as.integer(n_cells),
    group_label = as.character(group_label)
  )

refined_stats <- read_table(REFINED_STATS) %>%
  mutate(
    APO_median = num(APO_median),
    nonAPO_median = num(nonAPO_median),
    APO_mean = num(APO_mean),
    nonAPO_mean = num(nonAPO_mean),
    mean_difference = num(mean_difference),
    median_difference = num(median_difference),
    p_value = num(p_value),
    fdr_bh = num(fdr_bh),
    n_APO = as.integer(n_APO),
    n_nonAPO = as.integer(n_nonAPO)
  )

f_rows <- refined_scores %>%
  inner_join(f_specs, by = c("cell_scope", "module_name")) %>%
  left_join(
    refined_stats %>%
      select(
        cell_scope, module_name, object_name, APO_median, nonAPO_median, APO_mean, nonAPO_mean,
        mean_difference, median_difference, direction, p_value, fdr_bh, genes_found, n_genes_found,
        gene_coverage_flag, interpretability_flag, n_APO, n_nonAPO
      ),
    by = c("cell_scope", "module_name"),
    suffix = c("", "_stat")
  ) %>%
  transmute(
    priority,
    category,
    cell_scope,
    module_name,
    panel_label_plain = str_replace_all(panel_label, "\n", " | "),
    sample_id,
    group_label,
    score_value = module_score_mean,
    module_score_mean,
    module_score_median,
    percent_high_score_cells = NA_real_,
    n_cells,
    input_table = REFINED_SCORES,
    APO_median,
    nonAPO_median,
    APO_mean,
    nonAPO_mean,
    mean_difference,
    observed_direction = direction,
    nominal_p = p_value,
    fdr_bh,
    object_name,
    low_count,
    genes_found,
    n_genes_found,
    low_confidence_gene_coverage = gene_coverage_flag != "ok",
    expected_direction
  )

missing_cols <- setdiff(module_source_cols, names(f_rows))
for (cc in missing_cols) {
  f_rows[[cc]] <- NA
}
f_rows <- f_rows %>% select(all_of(module_source_cols))

source_out <- bind_rows(
  base_modules %>% select(all_of(module_source_cols)),
  f_rows
) %>%
  mutate(priority = as.integer(priority)) %>%
  arrange(priority, sample_id)

readr::write_tsv(source_out, OUT_SOURCE, na = "")

selected_data <- source_out %>%
  mutate(
    priority = as.integer(priority),
    score_value = num(score_value),
    nominal_p = num(nominal_p),
    fdr_bh = num(fdr_bh),
    APO_median = num(APO_median),
    nonAPO_median = num(nonAPO_median),
    APO_mean = num(APO_mean),
    nonAPO_mean = num(nonAPO_mean),
    mean_difference = num(mean_difference),
    group_label = factor(as.character(group_label), levels = c("non-APO", "APO"))
  ) %>%
  select(-any_of(c("panel_label", "panel_label_plain"))) %>%
  left_join(full_titles, by = "priority") %>%
  mutate(
    panel_label = factor(panel_label, levels = full_titles$panel_label),
    group_label = factor(as.character(group_label), levels = c("non-APO", "APO"))
  ) %>%
  filter(!is.na(score_value), !is.na(group_label), !is.na(panel_label))

if (n_distinct(selected_data$priority) != 10) {
  stop("Expected 10 modules, found ", n_distinct(selected_data$priority), ".")
}

stat_for_plot <- selected_data %>%
  distinct(
    priority, category, cell_scope, module_name, expected_direction, panel_label,
    APO_median, nonAPO_median, APO_mean, nonAPO_mean, mean_difference, observed_direction, nominal_p, fdr_bh
  ) %>%
  mutate(
    panel_label = factor(panel_label, levels = full_titles$panel_label),
    stat_label = paste0(star_label(nominal_p), "\n", format_p(nominal_p))
  ) %>%
  arrange(priority)

ypos <- selected_data %>%
  group_by(panel_label) %>%
  summarise(
    ymin = min(score_value, na.rm = TRUE),
    ymax = max(score_value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    raw_span = ymax - ymin,
    span = pmax(raw_span, abs(ymax) * 0.04, 0.035),
    y_lower = ymin - span * 0.045,
    y_upper = ymax + span * 0.165,
    line_y = ymax + span * 0.055,
    text_y = ymax + span * 0.115
  )

stat_for_plot <- stat_for_plot %>% left_join(ypos, by = "panel_label")
limit_data <- bind_rows(
  ypos %>% transmute(panel_label, group_label = factor("non-APO", levels = c("non-APO", "APO")), score_value = y_lower),
  ypos %>% transmute(panel_label, group_label = factor("non-APO", levels = c("non-APO", "APO")), score_value = y_upper)
)

p <- ggplot(selected_data, aes(x = group_label, y = score_value, color = group_label, fill = group_label)) +
  geom_blank(data = limit_data, aes(x = group_label, y = score_value), inherit.aes = FALSE) +
  geom_boxplot(width = 0.52, outlier.shape = NA, alpha = 0.18, linewidth = 0.32) +
  geom_jitter(width = 0.085, size = 0.78, alpha = 0.82, stroke = 0) +
  geom_segment(
    data = stat_for_plot,
    aes(x = 1, xend = 2, y = line_y, yend = line_y),
    inherit.aes = FALSE,
    linewidth = 0.22,
    color = "grey30"
  ) +
  geom_text(
    data = stat_for_plot,
    aes(x = 1.5, y = text_y, label = stat_label),
    inherit.aes = FALSE,
    size = 1.95,
    lineheight = 0.86,
    color = "black"
  ) +
  facet_wrap(~panel_label, nrow = 2, ncol = 5, scales = "free_y") +
  scale_color_manual(values = c("non-APO" = COL_NONAPO, "APO" = COL_APO), drop = FALSE) +
  scale_fill_manual(values = c("non-APO" = COL_NONAPO, "APO" = COL_APO), drop = FALSE) +
  scale_x_discrete(labels = c("non-APO" = "non-APO", "APO" = "APO")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) +
  labs(x = NULL, y = "Module score") +
  coord_cartesian(clip = "off") +
  theme_dotbox() +
  theme(panel.spacing.x = unit(1.6, "mm"), panel.spacing.y = unit(2.8, "mm"))

width_in <- 180 / 25.4
height_in <- 82 / 25.4
ggsave(OUT_PNG, p, width = width_in, height = height_in, units = "in", dpi = 600, limitsize = FALSE, bg = "white")

grDevices::pdf(
  OUT_PDF,
  width = width_in,
  height = height_in,
  onefile = FALSE,
  family = "Helvetica",
  useDingbats = FALSE,
  compress = FALSE,
  bg = "white"
)
print(p)
dev.off()

svg_device <- if (requireNamespace("svglite", quietly = TRUE)) {
  "svglite::svglite"
} else {
  "grDevices::svg"
}
if (requireNamespace("svglite", quietly = TRUE)) {
  svglite::svglite(OUT_SVG, width = width_in, height = height_in, bg = "white")
  print(p)
  dev.off()
} else {
  grDevices::svg(OUT_SVG, width = width_in, height = height_in, bg = "white", onefile = FALSE)
  print(p)
  dev.off()
}

stats_summary <- stat_for_plot %>%
  mutate(panel_label_plain = str_replace_all(as.character(panel_label), "\n", " | ")) %>%
  left_join(
    refined_stats %>%
      transmute(
        cell_scope, module_name,
        refined_direction = direction,
        refined_p_value = p_value,
        refined_fdr_bh = fdr_bh,
        refined_n_APO = n_APO,
        refined_n_nonAPO = n_nonAPO,
        refined_gene_coverage_flag = gene_coverage_flag,
        refined_interpretability_flag = interpretability_flag
      ),
    by = c("cell_scope", "module_name")
  ) %>%
  mutate(
    source_table = if_else(priority <= 7, "module_source", "Blineage_refinement")
  ) %>%
  select(
    priority, panel_label_plain, category, cell_scope, module_name, source_table,
    APO_median, nonAPO_median, APO_mean, nonAPO_mean, mean_difference, observed_direction,
    nominal_p, fdr_bh, expected_direction, starts_with("refined_")
  )
readr::write_tsv(stats_summary, OUT_STATS, na = "")

base_output <- source_out %>% filter(priority <= 7) %>% select(all_of(module_source_cols))
base_reference <- module_source %>% mutate(priority = as.integer(priority)) %>% filter(priority <= 7) %>% select(all_of(module_source_cols))
base_rows_match <- isTRUE(all.equal(base_output, base_reference, check.attributes = FALSE, tolerance = 0))
base_order_ok <- identical(sort(unique(base_output$priority)), 1:7)
f_stats <- stats_summary %>% filter(priority >= 8)
f_from_refinement <- all(f_stats$source_table == "Blineage_refinement")
f_apo_low <- all(f_stats$observed_direction == "APO_low")
f_p005 <- all(f_stats$nominal_p < 0.05)
outputs_exist <- all(file.exists(c(OUT_PNG, OUT_PDF, OUT_SVG, OUT_SOURCE, OUT_STATS, OUT_README)) | c(TRUE, TRUE, TRUE, TRUE, TRUE, FALSE))
png_dim <- read_png_dim(OUT_PNG)

module_lines <- stat_for_plot %>%
  arrange(priority) %>%
  mutate(line = paste0(
    priority, ". ", str_replace_all(as.character(panel_label), "\n", " | "),
    " | cell_scope=", cell_scope,
    " | module_name=", module_name,
    " | APO median=", fmt_num(APO_median, 4),
    " | non-APO median=", fmt_num(nonAPO_median, 4),
    " | direction=", observed_direction,
    " | nominal p=", fmt_num(nominal_p, 6),
    " | FDR/q=", fmt_num(fdr_bh, 6)
  )) %>%
  pull(line)

readme <- c(
  "Single-cell module-score plots",
  paste0("Generated: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")),
  "",
  "Script:",
  SCRIPT_PATH,
  "",
  "Input tables:",
  paste0("- module source: ", MODULE_SOURCE),
  paste0("- B-lineage module scores: ", REFINED_SCORES),
  paste0("- B-lineage module statistics: ", REFINED_STATS),
  "",
  "Module interpretation:",
  "- The B-lineage panel focuses on the BAFF/APRIL receptor-ligand axis.",
  "- Plasmablast B cells show decreased BAFF/APRIL receptor-response program.",
  "- FCGR3A/CD16 monocytes and CD14 classical monocytes show decreased BAFF/APRIL ligand-source modules.",
  "- Source-side modules reflect ligand-source transcriptional signal and should not be interpreted alone as direct cell-cell communication decrease.",
  "- Displayed p values are nominal Mann-Whitney U test p values from the source tables.",
  "",
  "Displayed modules:",
  module_lines,
  "",
  "Quality checks:",
  paste0("- base module rows match the source table: ", base_rows_match),
  paste0("- B-lineage rows come from the refinement tables: ", f_from_refinement),
  paste0("- B-lineage modules are all APO_low: ", f_apo_low),
  paste0("- B-lineage modules all have nominal p<0.05: ", f_p005),
  paste0("- base module order is valid: ", base_order_ok),
  paste0("- PDF generated: ", file.exists(OUT_PDF)),
  paste0("- PNG generated: ", file.exists(OUT_PNG)),
  paste0("- SVG generated: ", file.exists(OUT_SVG)),
  "",
  "Output files:",
  paste0("- PNG: ", OUT_PNG),
  paste0("- PDF: ", OUT_PDF),
  paste0("- SVG: ", OUT_SVG),
  paste0("- source_selected: ", OUT_SOURCE),
  paste0("- stats_summary: ", OUT_STATS),
  paste0("- README: ", OUT_README),
  "",
  paste0("PNG pixel dimensions: ", png_dim[["width"]], " x ", png_dim[["height"]]),
  "",
  "Layout notes:",
  "- One total figure only.",
  "- 2x5 layout.",
  "- No D/E/F labels and no large title were added.",
  "- White background.",
  "- non-APO and APO use fixed group colors.",
  paste0("- PDF output uses grDevices::pdf(compress=FALSE, useDingbats=FALSE); SVG output uses ", svg_device, ".")
)
writeLines(readme, OUT_README)

cat("Generated files:\n")
cat(paste(c(OUT_PNG, OUT_PDF, OUT_SVG, OUT_SOURCE, OUT_STATS, OUT_README), collapse = "\n"), "\n")
cat("Selected modules:\n")
cat(paste(module_lines, collapse = "\n"), "\n")
cat("QC base_rows_match:", base_rows_match, "\n")
cat("QC F_APO_low:", f_apo_low, "\n")
cat("QC F_p005:", f_p005, "\n")
cat("PNG dimensions:", png_dim[["width"]], "x", png_dim[["height"]], "\n")
cat("SVG device:", svg_device, "\n")
