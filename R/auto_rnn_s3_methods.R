# ================================================================
# AutoRNN Framework (torch) — S3 Methods Extension
# print.auto_rnn  ·  summary.auto_rnn  ·  plot.auto_rnn
# ================================================================
#
# SOURCE THIS FILE AFTER auto_rnn_framework.R:
#   source("auto_rnn_framework.R")
#   source("auto_rnn_s3_methods.R")
#
# Class hierarchy (set inside auto_rnn_torch()):
#   class(result) == c("auto_lstm", "auto_rnn")   for LSTM
#   class(result) == c("auto_gru",  "auto_rnn")   for GRU
#   class(result) == c("auto_rnn_v","auto_rnn")   for vanilla RNN
#
# All three inherit from "auto_rnn", so every method defined here
# works for auto_lstm(), auto_gru(), and auto_rnn() results.
# ================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)   # install.packages("patchwork")
  library(scales)      # install.packages("scales")
  library(tidyverse)
  library(cli)         # install.packages("cli")
})


# ================================================================
# INTERNAL HELPERS  (not exported)
# ================================================================

# ── Palette used consistently across all S3 methods ─────────────
.AUTO_RNN_PAL <- list(
  recursive = "#2166ac",
  direct    = "#d6604d",
  actual    = "grey25",
  train     = "#1b7837",
  val       = "#762a83",
  ci_rec    = "#2166ac",
  ci_dir    = "#d6604d",
  neutral   = "grey60",
  model     = c(LSTM = "#2166ac", GRU = "#4dac26", RNN = "#d6604d")
)

# ── Format a single number cleanly ──────────────────────────────
.fmt <- function(x, digits = 4) formatC(x, digits = digits,
                                         format = "f", flag = " ")

# ── Safe pull: returns NA if slot missing or NULL ────────────────
.slot <- function(result, name, default = NA) {
  v <- result[[name]]
  if (is.null(v)) default else v
}

# ── Pull CV summary mean for one metric string ───────────────────
.cv_mean <- function(result, metric_name) {
  smry <- result$cv_results$summary
  if (is.null(smry)) return(NA_real_)
  row  <- smry[smry$metric == metric_name, "mean", drop = TRUE]
  if (length(row) == 0L) NA_real_ else row[[1L]]
}

# ── Build a tidy forecast tibble combining both strategies ────────
.tidy_forecasts <- function(result) {
  fc_list <- Filter(Negate(is.null),
                    list(result$forecast_recursive,
                         result$forecast_direct))
  if (length(fc_list) == 0L) return(tibble())
  dplyr::bind_rows(fc_list)
}

# ── Determine which strategies were run ──────────────────────────
.strategies_run <- function(result) {
  s <- character(0)
  if (!is.null(result$forecast_recursive)) s <- c(s, "recursive")
  if (!is.null(result$forecast_direct))    s <- c(s, "direct")
  s
}

# ── Extract final training epoch count ───────────────────────────
.epochs_trained <- function(result) {
  h <- result$history_recursive
  if (is.null(h)) return(NA_integer_)
  nrow(h)
}

# ── Determine whether BO / CV was run ────────────────────────────
.ran_bo <- function(result) !is.null(result$bo_result)
.ran_cv <- function(result) !is.null(result$cv_results)

# ── Count direct models ───────────────────────────────────────────
.n_direct <- function(result) length(result$direct_models)

# ── Box-drawing line helper ───────────────────────────────────────
.box <- function(char = "═", width = 60) strrep(char, width)
.hdr <- function(txt, char = "─", width = 60) {
  pad <- max(0L, width - nchar(txt) - 4L)
  sprintf("%s  %s  %s", strrep(char, 2), txt, strrep(char, pad))
}


# ================================================================
# SECTION A — print.auto_rnn
# ================================================================

#' Print method for auto_rnn objects
#'
#' Displays a concise, well-formatted one-screen summary:
#'   • Model identity & architecture
#'   • Training configuration
#'   • Best hyperparameters (if BO was run)
#'   • Forecast horizon & forecast table (first/last rows)
#'   • CV aggregated metrics (if CV was run)
#'
#' @param x       Object of class "auto_rnn".
#' @param digits  Decimal places for numeric output.
#' @param n_fc    Max forecast rows to print per strategy.
#' @param ...     Ignored.
#'
#' @return Invisibly returns x.
#' @export
print.auto_rnn <- function(x, digits = 4L, n_fc = 6L, ...) {

  mt      <- toupper(x$model_type)
  strats  <- .strategies_run(x)
  n_obs   <- nrow(x$data)
  hp      <- x$best_params
  epochs  <- .epochs_trained(x)

  # ── Header ──────────────────────────────────────────────────────
  cat("\n", .box("═"), "\n", sep = "")
  cat(sprintf("  Auto%s  (torch)   —   Time Series Forecasting\n", mt))
  cat(.box("═"), "\n", sep = "")

  # ── Data ────────────────────────────────────────────────────────
  cat(.hdr("Data"), "\n")
  cat(sprintf("  Observations   : %d\n", n_obs))
  cat(sprintf("  Date range     : %s  →  %s\n",
              format(min(x$data$date)), format(max(x$data$date))))
  cat(sprintf("  Frequency      : %s\n",
              infer_frequency(x$data$date)))

  # ── Model configuration ─────────────────────────────────────────
  cat(.hdr("Model Configuration"), "\n")
  cat(sprintf("  Architecture   : %s\n", mt))
  cat(sprintf("  Strategies     : %s\n", paste(strats, collapse = " + ")))
  if (!is.null(hp)) {
    cat(sprintf("  hidden_size    : %d\n",   as.integer(hp$hidden_size)))
    cat(sprintf("  num_layers     : %d\n",   as.integer(hp$num_layers)))
    cat(sprintf("  dense_units    : %d\n",   as.integer(hp$dense_units)))
    cat(sprintf("  dropout        : %.3f\n", hp$dropout))
    cat(sprintf("  learning_rate  : %.2e\n", hp$lr))
    cat(sprintf("  batch_size     : %d\n",   as.integer(hp$batch_size)))
  }
  if (.ran_bo(x))
    cat(sprintf("  HP source      : Bayesian Optimisation ✓\n"))
  else
    cat(sprintf("  HP source      : Defaults\n"))

  # ── Training ─────────────────────────────────────────────────────
  cat(.hdr("Training"), "\n")
  if (!is.na(epochs))
    cat(sprintf("  Epochs trained : %d  (EarlyStopping active)\n", epochs))
  if ("direct" %in% strats && .n_direct(x) > 0L)
    cat(sprintf("  Direct models  : %d  (one per horizon step)\n",
                .n_direct(x)))

  # ── Forecasts ────────────────────────────────────────────────────
  fc_all <- .tidy_forecasts(x)
  if (nrow(fc_all) > 0L) {
    cat(.hdr("Forecast"), "\n")
    horizon <- max(table(fc_all$strategy))
    cat(sprintf("  Horizon        : %d steps\n", horizon))
    cat(sprintf("  CI             : 95%% empirical\n\n"))

    for (s in strats) {
      fc_s <- dplyr::filter(fc_all, strategy == s)
      cat(sprintf("  [%s]\n", toupper(s)))
      # Show first n_fc rows
      show <- head(fc_s, n_fc)
      print_df <- data.frame(
        date     = format(show$date),
        forecast = .fmt(show$forecast, digits),
        lower_95 = .fmt(show$lower_95, digits),
        upper_95 = .fmt(show$upper_95, digits)
      )
      print(print_df, row.names = FALSE, right = FALSE)
      if (nrow(fc_s) > n_fc)
        cat(sprintf("  … %d more rows\n", nrow(fc_s) - n_fc))
      cat("\n")
    }
  }

  # ── CV Metrics ───────────────────────────────────────────────────
  if (.ran_cv(x)) {
    cat(.hdr("Cross-Validation  (rolling_origin)"), "\n")
    smry <- x$cv_results$summary
    n_folds <- nrow(x$cv_results$cv_metrics)
    cat(sprintf("  Folds          : %d\n\n", n_folds))

    metrics <- c("RMSE","MAE","MAPE")
    header  <- sprintf("  %-22s %10s %10s\n", "Metric", "Mean", "SD")
    cat(header)
    cat(sprintf("  %s\n", strrep("-", 44)))

    for (s in strats) {
      cat(sprintf("  Strategy: %s\n", s))
      for (m in metrics) {
        key <- paste0(m, "_", s)
        row <- smry[smry$metric == key, ]
        if (nrow(row) > 0L)
          cat(sprintf("    %-20s %10s %10s\n", m,
                      .fmt(row$mean, digits),
                      .fmt(row$sd,   digits)))
      }
    }
  }

  cat(.box("═"), "\n\n", sep = "")
  invisible(x)
}


# ================================================================
# SECTION B — summary.auto_rnn
# ================================================================

#' Summary method for auto_rnn objects
#'
#' Returns (and prints) a structured S3 object of class
#' "summary.auto_rnn" containing:
#'   • $identity     — model type, data dimensions, date range
#'   • $config       — lags, horizon, strategies, BO/CV flags
#'   • $best_params  — optimal hyperparameters
#'   • $architecture — parameter counts (recursive model)
#'   • $training     — epochs, final train/val loss
#'   • $cv           — per-strategy CV metric table (mean ± sd)
#'   • $forecasts    — list of tidy tibbles per strategy
#'   • $residuals    — in-sample residual diagnostics
#'
#' @param object  Object of class "auto_rnn".
#' @param digits  Decimal places.
#' @param ...     Ignored.
#'
#' @return Object of class "summary.auto_rnn" (invisibly).
#' @export
summary.auto_rnn <- function(object, digits = 4L, ...) {

  mt     <- toupper(object$model_type)
  hp     <- object$best_params
  hist_r <- object$history_recursive

  # ── Identity ─────────────────────────────────────────────────────
  identity <- list(
    model_type   = mt,
    n_obs        = nrow(object$data),
    date_min     = min(object$data$date),
    date_max     = max(object$data$date),
    frequency    = infer_frequency(object$data$date),
    scaler_mean  = object$scaler$mean,
    scaler_sd    = object$scaler$sd
  )

  # ── Config ───────────────────────────────────────────────────────
  config <- list(
    strategies = .strategies_run(object),
    ran_bo     = .ran_bo(object),
    ran_cv     = .ran_cv(object),
    n_direct   = .n_direct(object)
  )

  # ── Best params ──────────────────────────────────────────────────
  best_params <- hp

  # ── Architecture: count trainable params in recursive model ──────
  arch <- list(total_params = NA_integer_, layers = NA_character_)
  if (!is.null(object$model_recursive)) {
    tryCatch({
      params <- sum(sapply(object$model_recursive$parameters,
                           function(p) p$numel()))
      arch$total_params <- params
      arch$layers <- sprintf(
        "%s(%d units × %d layers) → Linear(%d) → Linear(1)",
        mt, hp$hidden_size, hp$num_layers, hp$dense_units
      )
    }, error = function(e) NULL)
  }

  # ── Training diagnostics ─────────────────────────────────────────
  training <- list(
    epochs_trained = NA_integer_,
    final_train_loss = NA_real_,
    final_val_loss   = NA_real_,
    best_val_loss    = NA_real_,
    converged        = NA
  )
  if (!is.null(hist_r) && nrow(hist_r) > 0L) {
    training$epochs_trained  <- nrow(hist_r)
    training$final_train_loss <- tail(hist_r$train_loss, 1L)
    training$final_val_loss   <- tail(hist_r$val_loss, 1L)
    training$best_val_loss    <- min(hist_r$val_loss, na.rm = TRUE)
    # Converged = did not hit max epochs (early stopping fired)
    training$converged <- nrow(hist_r) < max(hist_r$epoch)
  }

  # ── CV metric table ───────────────────────────────────────────────
  cv_table <- NULL
  if (.ran_cv(object)) {
    smry <- object$cv_results$summary
    n_folds <- nrow(object$cv_results$cv_metrics)

    cv_table <- smry %>%
      mutate(
        strategy = dplyr::case_when(
          grepl("recursive", metric) ~ "recursive",
          grepl("direct",    metric) ~ "direct",
          TRUE                       ~ "unknown"
        ),
        metric_name = gsub("_(recursive|direct)$", "", metric),
        label = sprintf("%s ± %s",
                        formatC(mean, digits = digits, format = "f"),
                        formatC(sd,   digits = digits, format = "f"))
      ) %>%
      select(strategy, metric_name, mean, sd, label) %>%
      arrange(strategy, metric_name)

    attr(cv_table, "n_folds") <- n_folds
  }

  # ── Per-strategy forecast tibbles ────────────────────────────────
  forecasts <- list(
    recursive = object$forecast_recursive,
    direct    = object$forecast_direct
  )

  # ── In-sample residual diagnostics (recursive model) ─────────────
  residuals_diag <- NULL
  if (!is.null(object$model_recursive) && !is.null(hist_r)) {
    tryCatch({
      sv  <- make_supervised(
        scale_x(object$data$value, object$scaler),
        max(12L, as.integer(ceiling(
          nrow(if (!is.null(object$forecast_recursive))
                 object$forecast_recursive else object$forecast_direct) * 1.5
        )))
      )
      pred_sc  <- predict_rnn(object$model_recursive, sv$X, get_device())
      pred_raw <- unscale_x(pred_sc, object$scaler)
      act_raw  <- unscale_x(sv$y,    object$scaler)
      res      <- act_raw - pred_raw

      residuals_diag <- list(
        n         = length(res),
        mean      = mean(res),
        sd        = sd(res),
        min       = min(res),
        max       = max(res),
        skewness  = mean((res - mean(res))^3) / sd(res)^3,
        kurtosis  = mean((res - mean(res))^4) / sd(res)^4 - 3,
        ljung_box = tryCatch(
          Box.test(res, lag = 10L, type = "Ljung-Box"),
          error = function(e) NULL
        ),
        values    = res
      )
    }, error = function(e) NULL)
  }

  # ── Assemble summary object ────────────────────────────────────────
  out <- structure(
    list(
      identity      = identity,
      config        = config,
      best_params   = best_params,
      architecture  = arch,
      training      = training,
      cv_table      = cv_table,
      forecasts     = forecasts,
      residuals     = residuals_diag,
      digits        = digits
    ),
    class = "summary.auto_rnn"
  )

  print(out)
  invisible(out)
}


# ================================================================
# print method for summary.auto_rnn
# ================================================================

#' @export
print.summary.auto_rnn <- function(x, ...) {

  mt     <- x$identity$model_type
  digits <- x$digits
  W      <- 62L   # box width

  .line  <- function(w = W) cat(strrep("─", w), "\n", sep = "")
  .dline <- function(w = W) cat(strrep("═", w), "\n", sep = "")
  .sec   <- function(title) {
    cat(sprintf("  %-*s\n", W - 2L, title))
    .line()
  }

  # ── Title block ──────────────────────────────────────────────────
  .dline()
  cat(sprintf("  AUTO%s  ·  torch time-series forecasting\n", mt))
  .dline()

  # ── DATA ────────────────────────────────────────────────────────
  .sec("DATA")
  cat(sprintf("  %-22s %s  →  %s\n", "Date range:",
              format(x$identity$date_min),
              format(x$identity$date_max)))
  cat(sprintf("  %-22s %d\n", "Observations:", x$identity$n_obs))
  cat(sprintf("  %-22s %s\n", "Frequency:",    x$identity$frequency))
  cat(sprintf("  %-22s mean=%-10s sd=%s\n",
              "Scaler (z-score):",
              .fmt(x$identity$scaler_mean, digits),
              .fmt(x$identity$scaler_sd,   digits)))
  cat("\n")

  # ── CONFIGURATION ────────────────────────────────────────────────
  .sec("CONFIGURATION")
  cat(sprintf("  %-22s %s\n", "Architecture:", mt))
  cat(sprintf("  %-22s %s\n", "Strategies:",
              paste(x$config$strategies, collapse = "  +  ")))
  cat(sprintf("  %-22s %s\n", "Bayesian Optim:",
              if (x$config$ran_bo) "Yes ✓" else "No  (defaults used)"))
  cat(sprintf("  %-22s %s\n", "CV (rolling_origin):",
              if (x$config$ran_cv) "Yes ✓" else "No"))
  if ("direct" %in% x$config$strategies && x$config$n_direct > 0L)
    cat(sprintf("  %-22s %d\n", "Direct models:", x$config$n_direct))
  cat("\n")

  # ── BEST HYPERPARAMETERS ─────────────────────────────────────────
  if (!is.null(x$best_params)) {
    .sec("BEST HYPERPARAMETERS")
    hp <- x$best_params
    cat(sprintf("  %-22s %d\n",   "hidden_size:",  as.integer(hp$hidden_size)))
    cat(sprintf("  %-22s %d\n",   "num_layers:",   as.integer(hp$num_layers)))
    cat(sprintf("  %-22s %d\n",   "dense_units:",  as.integer(hp$dense_units)))
    cat(sprintf("  %-22s %.4f\n", "dropout:",      hp$dropout))
    cat(sprintf("  %-22s %.2e\n", "learning_rate:", hp$lr))
    cat(sprintf("  %-22s %d\n",   "batch_size:",   as.integer(hp$batch_size)))
    cat("\n")
  }

  # ── ARCHITECTURE ─────────────────────────────────────────────────
  if (!is.na(x$architecture$total_params)) {
    .sec("ARCHITECTURE")
    cat(sprintf("  %-22s %s\n", "Structure:", x$architecture$layers))
    cat(sprintf("  %-22s %s\n", "Trainable params:",
                format(x$architecture$total_params, big.mark = ",")))
    cat("\n")
  }

  # ── TRAINING ─────────────────────────────────────────────────────
  tr <- x$training
  if (!is.na(tr$epochs_trained)) {
    .sec("TRAINING  (recursive final model)")
    cat(sprintf("  %-22s %d\n", "Epochs trained:", tr$epochs_trained))
    cat(sprintf("  %-22s %s\n", "Final train loss:",
                .fmt(tr$final_train_loss, digits)))
    if (!is.na(tr$final_val_loss))
      cat(sprintf("  %-22s %s\n", "Final val loss:",
                  .fmt(tr$final_val_loss, digits)))
    if (!is.na(tr$best_val_loss))
      cat(sprintf("  %-22s %s\n", "Best val loss:",
                  .fmt(tr$best_val_loss, digits)))
    cat("\n")
  }

  # ── CROSS-VALIDATION ─────────────────────────────────────────────
  if (!is.null(x$cv_table)) {
    n_folds <- attr(x$cv_table, "n_folds")
    .sec(sprintf("CROSS-VALIDATION  (%d rolling-origin folds)", n_folds))
    cat(sprintf("  %-12s %-12s %14s %14s\n",
                "Strategy", "Metric", "Mean", "SD"))
    .line()
    prev_strat <- ""
    for (i in seq_len(nrow(x$cv_table))) {
      row <- x$cv_table[i, ]
      strat <- row$strategy
      if (strat != prev_strat) {
        cat(sprintf("  [%s]\n", toupper(strat)))
        prev_strat <- strat
      }
      cat(sprintf("    %-10s %-12s %14s %14s\n",
                  "", row$metric_name,
                  .fmt(row$mean, digits),
                  .fmt(row$sd,   digits)))
    }
    cat("\n")
  }

  # ── FORECAST PREVIEW ─────────────────────────────────────────────
  .sec("FORECAST PREVIEW")
  for (s in x$config$strategies) {
    fc <- x$forecasts[[s]]
    if (is.null(fc) || nrow(fc) == 0L) next
    cat(sprintf("  [%s]  (horizon = %d)\n", toupper(s), nrow(fc)))
    preview <- head(fc, 4L)
    cat(sprintf("    %-12s %12s %12s %12s\n",
                "Date", "Forecast", "Lower 95%", "Upper 95%"))
    for (j in seq_len(nrow(preview))) {
      cat(sprintf("    %-12s %12s %12s %12s\n",
                  format(preview$date[j]),
                  .fmt(preview$forecast[j],  digits),
                  .fmt(preview$lower_95[j],  digits),
                  .fmt(preview$upper_95[j],  digits)))
    }
    if (nrow(fc) > 4L) cat(sprintf("    … %d more rows\n", nrow(fc) - 4L))
    cat("\n")
  }

  # ── RESIDUAL DIAGNOSTICS ─────────────────────────────────────────
  if (!is.null(x$residuals)) {
    rd <- x$residuals
    .sec("IN-SAMPLE RESIDUAL DIAGNOSTICS  (recursive model)")
    cat(sprintf("  %-22s %d\n",   "n:", rd$n))
    cat(sprintf("  %-22s %s\n",   "Mean:", .fmt(rd$mean, digits)))
    cat(sprintf("  %-22s %s\n",   "SD:",   .fmt(rd$sd,   digits)))
    cat(sprintf("  %-22s %s  …  %s\n",
                "Range:",
                .fmt(rd$min, digits), .fmt(rd$max, digits)))
    cat(sprintf("  %-22s %s\n", "Skewness:", .fmt(rd$skewness, digits)))
    cat(sprintf("  %-22s %s\n", "Excess kurtosis:", .fmt(rd$kurtosis, digits)))
    if (!is.null(rd$ljung_box)) {
      lb <- rd$ljung_box
      cat(sprintf("  %-22s stat=%.4f  df=%d  p=%.4f  %s\n",
                  "Ljung-Box (lag 10):",
                  lb$statistic, lb$parameter, lb$p.value,
                  if (lb$p.value > 0.05) "✓ no autocorrelation"
                  else                   "⚠ autocorrelation detected"))
    }
    cat("\n")
  }

  .dline()
  invisible(x)
}


# ================================================================
# SECTION C — plot.auto_rnn
# ================================================================

#' Plot method for auto_rnn objects
#'
#' Generates a multi-panel diagnostic dashboard using {patchwork}.
#' Panels depend on what was computed:
#'
#'   Panel 1  [always]     Forecast plot — history + predictions + CI
#'   Panel 2  [always]     Training loss curves (train vs. validation)
#'   Panel 3  [if CV]      Rolling-origin CV metrics per fold
#'   Panel 4  [if model]   In-sample residuals over time
#'   Panel 5  [if model]   Residual histogram + normal overlay
#'   Panel 6  [if model]   Residual ACF (autocorrelation function)
#'
#' @param x         Object of class "auto_rnn".
#' @param which     Integer vector selecting panels (1–6). NULL = all.
#' @param last_n    Historical obs shown in panel 1 (NULL = all).
#' @param cv_metric Metric for CV panel: "RMSE" | "MAE" | "MAPE".
#' @param ncol      Columns in patchwork layout (NULL = auto).
#' @param title     Dashboard super-title (NULL = auto-generated).
#' @param theme_fn  A ggplot2 theme function (default: theme_minimal).
#' @param ...       Ignored.
#'
#' @return A patchwork object (invisibly).  The plot is also printed.
#' @export
plot.auto_rnn <- function(x,
                          which     = NULL,
                          last_n    = NULL,
                          cv_metric = "RMSE",
                          ncol      = NULL,
                          title     = NULL,
                          theme_fn  = theme_minimal,
                          ...) {

  mt     <- toupper(x$model_type)
  strats <- .strategies_run(x)
  pal    <- .AUTO_RNN_PAL
  base_theme <- theme_fn(base_size = 11) +
    theme(plot.title    = element_text(face = "bold", size = 11),
          plot.subtitle = element_text(size = 9, colour = "grey40"))

  # Decide which panels are available
  avail <- c(
    p1 = TRUE,                                   # Forecast
    p2 = !is.null(x$history_recursive),          # Training loss
    p3 = .ran_cv(x),                             # CV metrics
    p4 = !is.null(x$model_recursive),            # Residuals over time
    p5 = !is.null(x$model_recursive),            # Residual histogram
    p6 = !is.null(x$model_recursive)             # Residual ACF
  )

  if (is.null(which))
    which <- which(avail)
  else
    which <- intersect(which, which(avail))

  if (length(which) == 0L)
    stop("No available panels for the requested 'which' selection.")

  panels <- list()

  # ──────────────────────────────────────────────────────────────
  # PANEL 1 — Forecast + History + Confidence Bands
  # ──────────────────────────────────────────────────────────────
  if (1L %in% which) {
    hist_df <- x$data
    if (!is.null(last_n)) hist_df <- tail(hist_df, last_n)
    fc_all  <- .tidy_forecasts(x)

    p1 <- ggplot() +
      geom_line(data   = hist_df,
                aes(date, value),
                colour = pal$actual, linewidth = 0.8) +
      geom_vline(xintercept = as.numeric(max(hist_df$date)),
                 linetype = "dotted", colour = pal$neutral, linewidth = 0.6)

    if (nrow(fc_all) > 0L) {
      p1 <- p1 +
        geom_ribbon(data = fc_all,
                    aes(date, ymin = lower_95, ymax = upper_95,
                        fill = strategy),
                    alpha = 0.15) +
        geom_line(data  = fc_all,
                  aes(date, forecast, colour = strategy),
                  linewidth = 1.0, linetype = "dashed") +
        geom_point(data = fc_all,
                   aes(date, forecast, colour = strategy),
                   size = 2.2, shape = 21, fill = "white", stroke = 1.2) +
        scale_colour_manual(
          values = c(recursive = pal$recursive, direct = pal$direct),
          name   = "Strategy") +
        scale_fill_manual(
          values = c(recursive = pal$ci_rec, direct = pal$ci_dir),
          name   = "95% CI")
    }

    p1 <- p1 +
      labs(title    = sprintf("Auto%s — Forecast", mt),
           subtitle = sprintf("Horizon: %d steps  |  CI: 95%% empirical",
                              if (nrow(fc_all) > 0L)
                                max(table(fc_all$strategy)) else 0L),
           x = NULL, y = "Value") +
      base_theme +
      theme(legend.position = "top",
            legend.key.size = unit(0.4, "cm"))

    panels[["p1"]] <- p1
  }

  # ──────────────────────────────────────────────────────────────
  # PANEL 2 — Training Loss Curves
  # ──────────────────────────────────────────────────────────────
  if (2L %in% which && !is.null(x$history_recursive)) {
    h <- x$history_recursive

    loss_long <- h %>%
      select(epoch, Training = train_loss, Validation = val_loss) %>%
      pivot_longer(-epoch, names_to = "split", values_to = "MSE") %>%
      filter(!is.na(MSE))

    best_epoch <- h$epoch[which.min(replace(h$val_loss,
                                             is.na(h$val_loss), Inf))]

    p2 <- ggplot(loss_long, aes(epoch, MSE, colour = split)) +
      geom_vline(xintercept = best_epoch,
                 linetype = "dotted", colour = "grey60", linewidth = 0.6) +
      geom_line(linewidth = 0.85) +
      annotate("text", x = best_epoch, y = max(loss_long$MSE, na.rm = TRUE),
               label = sprintf("best\nepoch %d", best_epoch),
               hjust = -0.1, vjust = 1, size = 2.8, colour = "grey45") +
      scale_colour_manual(
        values = c(Training = pal$train, Validation = pal$val),
        name   = NULL) +
      scale_y_continuous(labels = scales::label_number(accuracy = 0.0001)) +
      labs(title    = "Training Loss",
           subtitle = "EarlyStopping active on validation loss",
           x = "Epoch", y = "MSE") +
      base_theme +
      theme(legend.position = "top",
            legend.key.size = unit(0.4, "cm"))

    panels[["p2"]] <- p2
  }

  # ──────────────────────────────────────────────────────────────
  # PANEL 3 — Rolling-Origin CV Metrics
  # ──────────────────────────────────────────────────────────────
  if (3L %in% which && .ran_cv(x)) {
    cv   <- x$cv_results$cv_metrics
    cols <- grep(cv_metric, names(cv), value = TRUE)

    if (length(cols) > 0L) {
      cv_long <- cv %>%
        select(fold, all_of(cols)) %>%
        pivot_longer(-fold, names_to = "strategy", values_to = "value") %>%
        mutate(strategy = gsub(paste0(cv_metric, "_"), "", strategy)) %>%
        filter(!is.na(value))

      # Compute mean line per strategy
      cv_means <- cv_long %>%
        group_by(strategy) %>%
        summarise(mean_val = mean(value), .groups = "drop")

      p3 <- ggplot(cv_long, aes(fold, value, colour = strategy)) +
        geom_hline(data = cv_means,
                   aes(yintercept = mean_val, colour = strategy),
                   linetype = "dashed", linewidth = 0.5, alpha = 0.6) +
        geom_line(linewidth = 0.75) +
        geom_point(size = 2.0, shape = 21, fill = "white", stroke = 1.1) +
        scale_colour_manual(
          values = c(recursive = pal$recursive, direct = pal$direct),
          name   = "Strategy") +
        scale_x_continuous(breaks = scales::breaks_pretty()) +
        labs(title    = sprintf("CV: %s per Fold", cv_metric),
             subtitle = sprintf("rolling_origin | %d folds | dashed = mean",
                                nrow(cv)),
             x = "Fold", y = cv_metric) +
        base_theme +
        theme(legend.position = "top",
              legend.key.size = unit(0.4, "cm"))

      panels[["p3"]] <- p3
    }
  }

  # ── Helper: compute in-sample residuals ─────────────────────────
  .get_residuals <- function() {
    tryCatch({
      horizon_n <- if (!is.null(x$forecast_recursive))
        nrow(x$forecast_recursive)
      else
        nrow(x$forecast_direct)
      lags_approx <- max(12L, as.integer(ceiling(horizon_n * 1.5)))
      sv     <- make_supervised(scale_x(x$data$value, x$scaler), lags_approx)
      pred   <- predict_rnn(x$model_recursive, sv$X, get_device())
      res    <- unscale_x(sv$y, x$scaler) - unscale_x(pred, x$scaler)
      dates  <- x$data$date[(lags_approx + 1L):nrow(x$data)]
      tibble(date = dates, residual = res)
    }, error = function(e) NULL)
  }

  residuals_df <- if (any(c(4L, 5L, 6L) %in% which) &&
                      !is.null(x$model_recursive))
    .get_residuals() else NULL

  # ──────────────────────────────────────────────────────────────
  # PANEL 4 — Residuals Over Time
  # ──────────────────────────────────────────────────────────────
  if (4L %in% which && !is.null(residuals_df)) {
    res_sd <- sd(residuals_df$residual)

    p4 <- ggplot(residuals_df, aes(date, residual)) +
      geom_hline(yintercept = 0,
                 colour = pal$neutral, linewidth = 0.6) +
      geom_hline(yintercept = c(-2 * res_sd, 2 * res_sd),
                 linetype = "dashed", colour = "#d6604d", linewidth = 0.5,
                 alpha = 0.7) +
      geom_point(colour = pal$recursive, alpha = 0.55, size = 1.2) +
      geom_smooth(method = "loess", formula = y ~ x, se = FALSE,
                  colour = "#d6604d", linewidth = 0.75, linetype = "solid") +
      annotate("text",
               x = max(residuals_df$date),
               y = c(2 * res_sd, -2 * res_sd),
               label = c("+2σ", "−2σ"),
               hjust = 1.1, vjust = -0.3, size = 2.8, colour = "#d6604d") +
      scale_y_continuous(labels = scales::label_number()) +
      labs(title    = "In-Sample Residuals",
           subtitle = "Red dashed = ±2σ  |  curve = LOESS trend",
           x = NULL, y = "Residual") +
      base_theme

    panels[["p4"]] <- p4
  }

  # ──────────────────────────────────────────────────────────────
  # PANEL 5 — Residual Histogram + Normal Overlay
  # ──────────────────────────────────────────────────────────────
  if (5L %in% which && !is.null(residuals_df)) {
    res    <- residuals_df$residual
    res_mu <- mean(res);  res_sd <- sd(res)

    p5 <- ggplot(residuals_df, aes(residual)) +
      geom_histogram(aes(y = after_stat(density)),
                     bins = 25L,
                     fill = pal$recursive, colour = "white",
                     alpha = 0.70) +
      stat_function(
        fun  = dnorm,
        args = list(mean = res_mu, sd = res_sd),
        colour = "#d6604d", linewidth = 0.9, linetype = "solid"
      ) +
      geom_vline(xintercept = res_mu,
                 colour = "grey30", linetype = "dashed", linewidth = 0.6) +
      labs(title    = "Residual Distribution",
           subtitle = sprintf("μ=%.4f  σ=%.4f  |  red = N(μ,σ²) overlay",
                              res_mu, res_sd),
           x = "Residual", y = "Density") +
      base_theme

    panels[["p5"]] <- p5
  }

  # ──────────────────────────────────────────────────────────────
  # PANEL 6 — Residual ACF
  # ──────────────────────────────────────────────────────────────
  if (6L %in% which && !is.null(residuals_df)) {
    res    <- residuals_df$residual
    n_lag  <- min(30L, floor(length(res) / 2L))
    acf_obj <- acf(res, lag.max = n_lag, plot = FALSE)
    ci_line <- qnorm(0.975) / sqrt(length(res))

    acf_df <- tibble(
      lag  = as.numeric(acf_obj$lag[-1]),    # drop lag 0
      acf  = as.numeric(acf_obj$acf[-1])
    )

    p6 <- ggplot(acf_df, aes(lag, acf)) +
      geom_hline(yintercept = 0, colour = "grey30", linewidth = 0.5) +
      geom_hline(yintercept = c(-ci_line, ci_line),
                 linetype = "dashed", colour = "#d6604d", linewidth = 0.6) +
      geom_segment(aes(xend = lag, yend = 0,
                       colour = abs(acf) > ci_line),
                   linewidth = 1.0, lineend = "butt") +
      geom_point(aes(colour = abs(acf) > ci_line), size = 1.8) +
      scale_colour_manual(
        values = c("FALSE" = pal$recursive, "TRUE" = "#d6604d"),
        guide  = "none") +
      scale_x_continuous(breaks = scales::breaks_pretty(n = 6)) +
      labs(title    = "Residual ACF",
           subtitle = "Red dashed = 95% CI  |  red bars = significant lags",
           x = "Lag", y = "ACF") +
      base_theme

    panels[["p6"]] <- p6
  }

  # ── Assemble with patchwork ──────────────────────────────────────
  n_panels <- length(panels)
  if (is.null(ncol))
    ncol <- if (n_panels <= 2L) n_panels
            else if (n_panels <= 4L) 2L
            else 3L

  if (is.null(title))
    title <- sprintf(
      "Auto%s (torch) Diagnostic Dashboard  ·  %d obs  ·  %s",
      mt, nrow(x$data),
      paste(strats, collapse = " + ")
    )

  dashboard <- patchwork::wrap_plots(panels, ncol = ncol) +
    patchwork::plot_annotation(
      title   = title,
      theme   = theme(
        plot.title = element_text(
          face = "bold", size = 14, hjust = 0.5,
          margin = margin(b = 8)
        )
      )
    )

  print(dashboard)
  invisible(dashboard)
}


# ================================================================
# SECTION D — format.auto_rnn  (one-liner banner)
# ================================================================

#' @export
format.auto_rnn <- function(x, ...) {
  mt     <- toupper(x$model_type)
  strats <- paste(.strategies_run(x), collapse = "+")
  hp     <- x$best_params
  bo_tag <- if (.ran_bo(x)) "BO" else "default-HP"
  cv_tag <- if (.ran_cv(x)) "CV" else "no-CV"
  sprintf("<auto_%s | n=%d | h=%d | %s | %s | %s>",
          tolower(mt), nrow(x$data),
          if (!is.null(x$forecast_recursive))
            nrow(x$forecast_recursive)
          else if (!is.null(x$forecast_direct))
            nrow(x$forecast_direct)
          else NA,
          strats, bo_tag, cv_tag)
}


# ================================================================
# SECTION E — coef.auto_rnn  (return best hyperparameters)
# ================================================================

#' Return best hyperparameters as a named numeric vector
#'
#' @param object  Object of class "auto_rnn".
#' @param ...     Ignored.
#' @return Named numeric vector.
#' @export
coef.auto_rnn <- function(object, ...) {
  hp <- object$best_params
  if (is.null(hp)) return(NULL)
  unlist(lapply(hp, as.numeric))
}


# ================================================================
# SECTION F — fitted.auto_rnn  (in-sample fitted values)
# ================================================================

#' Extract in-sample fitted values from the recursive model
#'
#' @param object  Object of class "auto_rnn".
#' @param ...     Ignored.
#' @return Tibble with columns: date, fitted, actual, residual.
#' @export
fitted.auto_rnn <- function(object, ...) {
  if (is.null(object$model_recursive))
    stop("No recursive model found. Run with strategy = 'recursive' or 'both'.")

  horizon_n   <- if (!is.null(object$forecast_recursive))
    nrow(object$forecast_recursive) else nrow(object$forecast_direct)
  lags_approx <- max(12L, as.integer(ceiling(horizon_n * 1.5)))

  sv     <- make_supervised(scale_x(object$data$value, object$scaler),
                            lags_approx)
  pred   <- predict_rnn(object$model_recursive, sv$X, get_device())
  fitted_vals <- unscale_x(pred,   object$scaler)
  actual_vals <- unscale_x(sv$y,   object$scaler)

  tibble(
    date     = object$data$date[(lags_approx + 1L):nrow(object$data)],
    fitted   = fitted_vals,
    actual   = actual_vals,
    residual = actual_vals - fitted_vals
  )
}


# ================================================================
# SECTION G — residuals.auto_rnn
# ================================================================

#' Extract in-sample residuals from the recursive model
#'
#' @param object  Object of class "auto_rnn".
#' @param ...     Ignored.
#' @return Numeric vector of residuals.
#' @export
residuals.auto_rnn <- function(object, ...) {
  fitted(object)$residual
}


# ================================================================
# SECTION H — autoplot.auto_rnn  (ggplot2 autoplot interface)
# ================================================================

#' ggplot2 autoplot interface for auto_rnn objects
#'
#' Equivalent to plot.auto_rnn() but returns a patchwork object
#' without printing — useful for saving to files.
#'
#' @param object  Object of class "auto_rnn".
#' @param ...     Forwarded to plot.auto_rnn().
#'
#' @return A patchwork object (not printed).
#' @export
autoplot.auto_rnn <- function(object, ...) {
  p <- plot.auto_rnn(object, ...)
  invisible(p)
}


# ================================================================
# SECTION I — EXAMPLE USAGE
# ================================================================

# ── Uncomment to run ─────────────────────────────────────────────
#
# source("auto_rnn_framework.R")   # load the main framework first
# library(zoo)
#
# df <- data.frame(
#   date  = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
#   value = as.numeric(AirPassengers)
# )
#
# result <- auto_lstm(
#   data = df, horizon = 12L, strategy = "both",
#   run_bo = TRUE, bo_init = 5L, bo_iter = 10L,
#   run_cv = TRUE, cv_initial = 84L, verbose = 1L
# )
#
# # ── S3 generics ─────────────────────────────────────────────────
#
# print(result)              # concise dashboard to console
#
# sm <- summary(result)      # full structured summary + print
# sm$cv_table                # access the CV metric tibble
# sm$residuals$ljung_box     # access Ljung-Box test result
# sm$training$best_val_loss  # access best validation loss
#
# plot(result)                         # full 6-panel dashboard
# plot(result, which = c(1, 2))        # only forecast + loss
# plot(result, which = c(4, 5, 6))     # only residual panels
# plot(result, last_n = 60, ncol = 2)  # last 60 obs, 2-col layout
# plot(result, cv_metric = "MAPE")     # CV panel uses MAPE
#
# # ── Other generics ───────────────────────────────────────────────
# format(result)          # one-line banner string
# coef(result)            # best hyperparameters as named vector
# fv <- fitted(result)    # tibble: date, fitted, actual, residual
# rs <- residuals(result) # numeric residual vector
#
# # ── Save dashboard to file ───────────────────────────────────────
# library(ggplot2)
# p <- autoplot(result)
# ggsave("auto_lstm_dashboard.png", p, width = 14, height = 10, dpi = 150)
