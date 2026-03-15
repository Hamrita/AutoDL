# ================================================================
# AutoRNN Framework — Extended S3 Methods & Statistical Models
# ================================================================
#
# This file extends auto_rnn_s3_methods.R with:
#
#   PART A — Out-of-sample fitted values
#     fitted_oos.auto_rnn()     full OOS fitted tibble per fold
#
#   PART B — predict() method for both strategies
#     predict.auto_rnn()        recursive | direct | both
#
#   PART C — predicted() extractor (OOS actual vs predicted)
#     predicted.auto_rnn()      per-fold actuals + predictions
#
#   PART D — OOS CV performance evaluation
#     cv_performance()          detailed per-fold + aggregate table
#     plot_cv_performance()     multi-metric diagnostic plot
#
#   PART E — Auto-ARIMA model
#     auto_arima_ts()           auto.arima wrapper with full pipeline
#     print / summary / predict / fitted / residuals / plot
#
#   PART F — Auto-ARFIMA model
#     auto_arfima_ts()          auto.arfima (long-memory) wrapper
#     print / summary / predict / fitted / residuals / plot
#
#   PART G — Unified model comparison
#     compare_all_models()      RNN + ARIMA + ARFIMA on same data
#
# ================================================================
# Additional packages required:
#   install.packages(c(
#     "forecast",   # auto.arima, auto.arfima, Arima
#     "fracdiff",   # ARFIMA estimation
#     "patchwork",  # multi-panel plots
#     "scales",     # axis formatting
#     "tseries"     # adf.test for stationarity
#   ))
# ================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(rsample)
  library(patchwork)
  library(scales)
  library(forecast)
  library(fracdiff)
  library(tseries)
  library(zoo)
})


# ================================================================
# SHARED INTERNAL HELPERS
# ================================================================

.fmt      <- function(x, d = 4) formatC(x, digits = d, format = "f", flag = " ")
.box      <- function(ch = "═", w = 62) strrep(ch, w)
.hdr      <- function(txt, ch = "─", w = 62) {
  pad <- max(0L, w - nchar(txt) - 4L)
  sprintf("%s  %s  %s", strrep(ch, 2), txt, strrep(ch, pad))
}

.PAL <- list(
  recursive = "#2166ac", direct    = "#d6604d",
  arima     = "#1b7837", arfima    = "#8c510a",
  actual    = "grey20",  neutral   = "grey60",
  train     = "#35978f", val       = "#bf812d",
  ci_lo     = "#92c5de", ci_hi     = "#f4a582"
)

# Safe metric calculation (handles edge cases)
.safe_metrics <- function(actual, predicted) {
  r    <- actual - predicted
  mape <- if (any(actual == 0)) NA_real_
          else mean(abs(r / actual)) * 100
  list(
    RMSE  = sqrt(mean(r^2, na.rm = TRUE)),
    MAE   = mean(abs(r),   na.rm = TRUE),
    MAPE  = mape,
    SMAPE = mean(200 * abs(r) / (abs(actual) + abs(predicted)),
                 na.rm = TRUE),
    MBE   = mean(r, na.rm = TRUE)
  )
}

# Infer frequency integer from date vector (for ts() objects)
.ts_freq <- function(dates) {
  med <- median(as.numeric(diff(sort(dates))))
  dplyr::case_when(
    med <= 1  ~ 365L,
    med <= 8  ~ 52L,
    med <= 32 ~ 12L,
    med <= 93 ~ 4L,
    TRUE      ~ 1L
  )
}

# Convert data frame to ts object
.df_to_ts <- function(df) {
  freq  <- .ts_freq(df$date)
  start <- c(as.integer(format(min(df$date), "%Y")),
             switch(as.character(freq),
                    "12" = as.integer(format(min(df$date), "%m")),
                    "4"  = as.integer(ceiling(
                      as.integer(format(min(df$date), "%m")) / 3)),
                    "52" = as.integer(format(min(df$date), "%W")),
                    1L))
  ts(df$value, start = start, frequency = freq)
}

# Strategies run helper
.strategies_run <- function(x) {
  s <- character(0)
  if (!is.null(x$forecast_recursive)) s <- c(s, "recursive")
  if (!is.null(x$forecast_direct))    s <- c(s, "direct")
  s
}


# ================================================================
# PART A — OUT-OF-SAMPLE FITTED VALUES  (fitted_oos)
# ================================================================

#' Extract out-of-sample fitted values from rolling_origin CV
#'
#' For each rolling-origin fold, returns the model's predictions
#' on the held-out assessment window alongside the true values.
#' Unlike in-sample fitted(), these predictions were NEVER seen
#' during that fold's training — they are genuinely out-of-sample.
#'
#' @param object    Object of class "auto_rnn".
#' @param strategy  "recursive" | "direct" | "both".
#' @param ...       Ignored.
#'
#' @return Tibble with columns:
#'   fold, h (horizon step), date, actual, predicted,
#'   residual, strategy
#'
#' @export
fitted_oos <- function(object, ...) UseMethod("fitted_oos")

#' @export
fitted_oos.auto_rnn <- function(object,
                                strategy = c("both","recursive","direct"),
                                ...) {
  strategy <- match.arg(strategy)

  if (is.null(object$cv_results))
    stop("No CV results found. Re-run with run_cv = TRUE.")

  folds   <- object$cv_results$fold_forecasts
  horizon <- if (!is.null(object$forecast_recursive))
               nrow(object$forecast_recursive)
             else nrow(object$forecast_direct)

  freq  <- infer_frequency(object$data$date)
  rows  <- list()

  for (fd in folds) {
    fold_id <- fd$fold
    actuals <- fd$actual
    n_act   <- length(actuals)

    # Recover assessment dates
    # Train uses first n_train obs; assessment follows immediately
    n_tr     <- fd$n_train
    all_dates <- object$data$date
    assess_start_idx <- n_tr + 1L
    assess_end_idx   <- min(n_tr + n_act, length(all_dates))
    assess_dates <- if (assess_end_idx <= length(all_dates))
      all_dates[assess_start_idx:assess_end_idx]
    else
      seq(all_dates[length(all_dates)], by = freq,
          length.out = n_act + 1L)[-1L]

    for (s in c("recursive","direct")) {
      if (strategy == "both" || strategy == s) {
        fc_name <- paste0(s, "_forecast")
        preds   <- fd[[fc_name]]
        if (is.null(preds)) next

        preds_trim <- preds[seq_len(n_act)]
        n_use      <- min(n_act, length(assess_dates))

        rows[[length(rows) + 1L]] <- tibble(
          fold      = fold_id,
          h         = seq_len(n_use),
          date      = assess_dates[seq_len(n_use)],
          actual    = actuals[seq_len(n_use)],
          predicted = preds_trim[seq_len(n_use)],
          residual  = actuals[seq_len(n_use)] - preds_trim[seq_len(n_use)],
          strategy  = s
        )
      }
    }
  }

  if (length(rows) == 0L)
    stop("No matching OOS predictions found for the requested strategy.")

  dplyr::bind_rows(rows)
}


# ================================================================
# PART B — predict() METHOD
# ================================================================

#' Predict method for auto_rnn objects
#'
#' Generates forecasts for a new data window or extends the
#' existing forecast. Can also re-forecast with a custom window.
#'
#' @param object      Object of class "auto_rnn".
#' @param newdata     Optional: data frame with `date` + `value` columns
#'                    providing a new observation window. If NULL,
#'                    uses the original training series tail.
#' @param horizon     Steps ahead (NULL = use original horizon).
#' @param strategy    "recursive" | "direct" | "both".
#' @param level       Confidence level for prediction intervals (0–100).
#' @param ...         Ignored.
#'
#' @return Tibble with columns:
#'   date, predicted, lower_{level}, upper_{level}, strategy
#'
#' @export
predict.auto_rnn <- function(object,
                             newdata  = NULL,
                             horizon  = NULL,
                             strategy = c("both","recursive","direct"),
                             level    = 95,
                             ...) {
  strategy <- match.arg(strategy)
  if (is.null(horizon))
    horizon <- if (!is.null(object$forecast_recursive))
                 nrow(object$forecast_recursive)
               else nrow(object$forecast_direct)

  # Determine input series
  if (!is.null(newdata)) {
    stopifnot(is.data.frame(newdata),
              "date"  %in% names(newdata),
              "value" %in% names(newdata))
    input_vals <- newdata$value
    last_date  <- max(newdata$date)
    freq       <- infer_frequency(newdata$date)
  } else {
    input_vals <- object$data$value
    last_date  <- max(object$data$date)
    freq       <- infer_frequency(object$data$date)
  }

  sc  <- object$scaler
  scaled <- scale_x(input_vals, sc)

  # Compute lags from existing model or infer
  horizon_orig <- if (!is.null(object$forecast_recursive))
    nrow(object$forecast_recursive) else nrow(object$forecast_direct)
  lags <- max(12L, as.integer(ceiling(horizon_orig * 1.5)))
  last_win <- tail(scaled, lags)

  # Empirical residual SD (from in-sample recursive fit)
  res_sd <- tryCatch({
    sv   <- make_supervised(scale_x(object$data$value, sc), lags)
    pred <- predict_rnn(object$model_recursive %||%
                          object$direct_models[[1]], sv$X, get_device())
    sd(unscale_x(sv$y, sc) - unscale_x(pred, sc))
  }, error = function(e) sd(input_vals) * 0.1)

  z_val     <- qnorm(0.5 + level / 200)
  fut_dates <- seq(last_date, by = freq, length.out = horizon + 1L)[-1L]
  ci_width  <- z_val * res_sd * sqrt(seq_len(horizon))

  lname_lo <- paste0("lower_", level)
  lname_hi <- paste0("upper_", level)

  results <- list()

  # ── Recursive prediction ────────────────────────────────────────
  if (strategy %in% c("recursive","both") &&
      !is.null(object$model_recursive)) {
    fc <- recursive_forecast(object$model_recursive, last_win, horizon, sc,
                             get_device())
    results[["recursive"]] <- tibble(
      date             = fut_dates,
      predicted        = fc,
      !!lname_lo      := fc - ci_width,
      !!lname_hi      := fc + ci_width,
      strategy         = "recursive"
    )
  }

  # ── Direct prediction ───────────────────────────────────────────
  if (strategy %in% c("direct","both") &&
      !is.null(object$direct_models) &&
      length(object$direct_models) >= horizon) {
    dm <- object$direct_models[seq_len(horizon)]
    fc <- direct_forecast(dm, last_win, sc, get_device())
    results[["direct"]] <- tibble(
      date             = fut_dates,
      predicted        = fc,
      !!lname_lo      := fc - ci_width,
      !!lname_hi      := fc + ci_width,
      strategy         = "direct"
    )
  }

  if (length(results) == 0L)
    stop("No models available for the requested strategy.")

  dplyr::bind_rows(results)
}

# Pipe-safe NULL coalescing
`%||%` <- function(a, b) if (!is.null(a)) a else b


# ================================================================
# PART C — predicted() EXTRACTOR  (OOS actual vs predicted)
# ================================================================

#' Extract out-of-sample actual and predicted values
#'
#' Returns a tidy tibble of all OOS predictions from
#' rolling_origin CV, paired with their true values.
#' Useful for custom metric computation and plotting.
#'
#' @param object    Object of class "auto_rnn".
#' @param strategy  "recursive" | "direct" | "both".
#' @param aggregate If TRUE, aggregate across folds (mean per h-step).
#' @param ...       Ignored.
#'
#' @return Tibble: fold, h, date, actual, predicted, residual, strategy
#'         (or aggregated version with mean_actual, mean_predicted, etc.)
#'
#' @export
predicted <- function(object, ...) UseMethod("predicted")

#' @export
predicted.auto_rnn <- function(object,
                               strategy  = c("both","recursive","direct"),
                               aggregate = FALSE,
                               ...) {
  oos <- fitted_oos(object, strategy = strategy)

  if (!aggregate) return(oos)

  # Aggregate across folds per horizon step
  oos %>%
    group_by(h, strategy) %>%
    summarise(
      n_folds         = dplyr::n(),
      mean_actual     = mean(actual,    na.rm = TRUE),
      mean_predicted  = mean(predicted, na.rm = TRUE),
      mean_residual   = mean(residual,  na.rm = TRUE),
      sd_residual     = sd(residual,    na.rm = TRUE),
      RMSE            = sqrt(mean(residual^2, na.rm = TRUE)),
      MAE             = mean(abs(residual),   na.rm = TRUE),
      .groups         = "drop"
    ) %>%
    arrange(strategy, h)
}


# ================================================================
# PART D — CV PERFORMANCE EVALUATION
# ================================================================

#' Detailed out-of-sample CV performance evaluation
#'
#' Computes comprehensive metrics per fold, per horizon step,
#' and aggregated — for both recursive and direct strategies.
#'
#' @param object     Object of class "auto_rnn".
#' @param strategy   "recursive" | "direct" | "both".
#' @param by_horizon If TRUE, also report metrics broken down by h step.
#' @param ...        Ignored.
#'
#' @return List:
#'   $per_fold    — tibble: fold × strategy × {RMSE,MAE,MAPE,SMAPE,MBE}
#'   $aggregate   — tibble: strategy × metric → mean ± sd
#'   $by_horizon  — (if by_horizon=TRUE) tibble: h × strategy × metrics
#'   $raw_oos     — full OOS tibble from fitted_oos()
#'
#' @export
cv_performance <- function(object, ...) UseMethod("cv_performance")

#' @export
cv_performance.auto_rnn <- function(object,
                                    strategy   = c("both","recursive","direct"),
                                    by_horizon = TRUE,
                                    ...) {
  strategy <- match.arg(strategy)
  oos      <- fitted_oos(object, strategy = strategy)

  # ── Per-fold metrics ────────────────────────────────────────────
  per_fold <- oos %>%
    group_by(fold, strategy) %>%
    summarise(
      n      = dplyr::n(),
      RMSE   = sqrt(mean(residual^2,       na.rm = TRUE)),
      MAE    = mean(abs(residual),          na.rm = TRUE),
      MAPE   = if (any(actual == 0)) NA_real_
               else mean(abs(residual / actual) * 100, na.rm = TRUE),
      SMAPE  = mean(200 * abs(residual) /
                      (abs(actual) + abs(predicted)), na.rm = TRUE),
      MBE    = mean(residual,               na.rm = TRUE),
      R2     = 1 - sum(residual^2) /
                   sum((actual - mean(actual))^2),
      .groups = "drop"
    ) %>%
    arrange(strategy, fold)

  # ── Aggregate (mean ± SD across folds) ─────────────────────────
  metric_cols <- c("RMSE","MAE","MAPE","SMAPE","MBE","R2")
  aggregate <- per_fold %>%
    group_by(strategy) %>%
    summarise(across(all_of(metric_cols),
                     list(mean = ~mean(.x, na.rm = TRUE),
                          sd   = ~sd(.x,   na.rm = TRUE),
                          min  = ~min(.x,  na.rm = TRUE),
                          max  = ~max(.x,  na.rm = TRUE)),
                     .names = "{.col}__{.fn}"),
              n_folds = dplyr::n(),
              .groups = "drop") %>%
    pivot_longer(-c(strategy, n_folds),
                 names_to  = c("metric","stat"),
                 names_sep = "__") %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    arrange(strategy, metric)

  # ── By-horizon metrics ──────────────────────────────────────────
  by_h <- NULL
  if (by_horizon) {
    by_h <- oos %>%
      group_by(h, strategy) %>%
      summarise(
        n_folds = dplyr::n(),
        RMSE    = sqrt(mean(residual^2,       na.rm = TRUE)),
        MAE     = mean(abs(residual),          na.rm = TRUE),
        MAPE    = if (any(actual == 0)) NA_real_
                  else mean(abs(residual / actual) * 100, na.rm = TRUE),
        MBE     = mean(residual,               na.rm = TRUE),
        .groups = "drop"
      ) %>%
      arrange(strategy, h)
  }

  # ── Print summary ───────────────────────────────────────────────
  mt <- toupper(object$model_type)
  cat("\n", .box("═"), "\n", sep = "")
  cat(sprintf("  [%s] Out-of-Sample CV Performance\n", mt))
  cat(.box("═"), "\n", sep = "")
  for (s in unique(per_fold$strategy)) {
    cat(.hdr(sprintf("Strategy: %s", toupper(s))), "\n")
    agg_s <- aggregate[aggregate$strategy == s, ]
    for (m in metric_cols) {
      r <- agg_s[agg_s$metric == m, ]
      if (nrow(r) == 0L) next
      cat(sprintf("  %-8s mean=%s  sd=%s  [%s, %s]\n",
                  m,
                  .fmt(r$mean, 4), .fmt(r$sd, 4),
                  .fmt(r$min, 4),  .fmt(r$max, 4)))
    }
    cat("\n")
  }
  cat(.box("═"), "\n\n", sep = "")

  structure(
    list(per_fold   = per_fold,
         aggregate  = aggregate,
         by_horizon = by_h,
         raw_oos    = oos),
    class = "cv_performance"
  )
}


#' Plot CV performance diagnostics
#'
#' @param x        Output of cv_performance().
#' @param metrics  Character vector of metrics to plot.
#' @param ncol     Layout columns.
#' @param ...      Ignored.
#'
#' @return patchwork object (invisibly).
#' @export
plot_cv_performance <- function(x, ...) UseMethod("plot_cv_performance")

#' @export
plot_cv_performance.cv_performance <- function(
    x,
    metrics = c("RMSE","MAE","MAPE"),
    ncol    = 2L,
    ...) {

  pal   <- .PAL
  btheme <- theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11))
  panels <- list()

  # P1: Metric per fold (faceted)
  pf_long <- x$per_fold %>%
    select(fold, strategy, all_of(metrics)) %>%
    pivot_longer(-c(fold, strategy),
                 names_to = "metric", values_to = "value")

  panels$fold_metrics <- ggplot(pf_long,
                                aes(fold, value, colour = strategy)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.8, shape = 21, fill = "white", stroke = 1) +
    facet_wrap(~metric, scales = "free_y", ncol = length(metrics)) +
    scale_colour_manual(
      values = c(recursive = pal$recursive, direct = pal$direct)) +
    labs(title = "OOS Metrics per Fold",
         x = "Fold", y = NULL, colour = "Strategy") +
    btheme + theme(legend.position = "top")

  # P2: Aggregate bar chart (mean ± SD)
  agg <- x$aggregate %>%
    filter(metric %in% metrics) %>%
    mutate(metric = factor(metric, levels = metrics))

  panels$agg_bar <- ggplot(agg, aes(strategy, mean, fill = strategy)) +
    geom_col(width = 0.55, show.legend = FALSE) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                  width = 0.2, colour = "grey30") +
    facet_wrap(~metric, scales = "free_y") +
    scale_fill_manual(
      values = c(recursive = pal$recursive, direct = pal$direct)) +
    labs(title = "Aggregate CV Metrics (mean ± SD)",
         x = NULL, y = "Value") +
    btheme

  # P3: By-horizon (if available)
  if (!is.null(x$by_horizon)) {
    bh_long <- x$by_horizon %>%
      select(h, strategy, all_of(intersect(metrics, names(x$by_horizon)))) %>%
      pivot_longer(-c(h, strategy),
                   names_to = "metric", values_to = "value")

    panels$by_h <- ggplot(bh_long,
                          aes(h, value, colour = strategy)) +
      geom_line(linewidth = 0.8) +
      geom_point(size = 1.6, shape = 21, fill = "white", stroke = 1) +
      facet_wrap(~metric, scales = "free_y") +
      scale_colour_manual(
        values = c(recursive = pal$recursive, direct = pal$direct)) +
      labs(title = "OOS Metrics by Horizon Step (h)",
           x = "Horizon Step h", y = NULL, colour = "Strategy") +
      btheme + theme(legend.position = "top")
  }

  # P4: OOS actual vs predicted scatter (per strategy)
  panels$scatter <- ggplot(x$raw_oos,
                           aes(actual, predicted, colour = strategy)) +
    geom_abline(slope = 1, intercept = 0,
                colour = "grey50", linetype = "dashed") +
    geom_point(alpha = 0.45, size = 1.4) +
    scale_colour_manual(
      values = c(recursive = pal$recursive, direct = pal$direct)) +
    labs(title = "OOS Actual vs Predicted",
         x = "Actual", y = "Predicted", colour = "Strategy") +
    coord_equal() + btheme + theme(legend.position = "top")

  # P5: OOS residual distribution
  panels$res_hist <- ggplot(x$raw_oos,
                            aes(residual, fill = strategy)) +
    geom_histogram(bins = 30L, alpha = 0.6, position = "identity",
                   colour = "white") +
    geom_vline(xintercept = 0, colour = "grey30",
               linetype = "dashed", linewidth = 0.7) +
    scale_fill_manual(
      values = c(recursive = pal$recursive, direct = pal$direct)) +
    labs(title = "OOS Residual Distribution",
         x = "Residual", y = "Count", fill = "Strategy") +
    btheme + theme(legend.position = "top")

  dashboard <- patchwork::wrap_plots(panels, ncol = ncol) +
    patchwork::plot_annotation(
      title = "Out-of-Sample CV Performance Dashboard",
      theme = theme(plot.title = element_text(
        face = "bold", size = 14, hjust = 0.5))
    )

  print(dashboard)
  invisible(dashboard)
}


# ================================================================
# PART E — AUTO-ARIMA MODEL
# ================================================================

#' Fit Auto-ARIMA with full pipeline (BO-free, uses AICc selection)
#'
#' Wraps forecast::auto.arima() with:
#'   • Automatic order selection via AICc
#'   • rolling_origin OOS cross-validation
#'   • Unified S3 class "auto_arima_ts" for print/summary/predict/plot
#'
#' @param data          Data frame with `date` + `value` columns.
#' @param date_col      Date column name.
#' @param value_col     Value column name.
#' @param horizon       Forecast steps ahead.
#' @param level         Prediction interval level(s): e.g. c(80, 95).
#' @param stepwise      Use stepwise ARIMA search (faster; less exhaustive).
#' @param approximation Use approximations for large datasets.
#' @param lambda        Box-Cox lambda (NULL = no transform; "auto" = estimate).
#' @param run_cv        Run rolling_origin OOS evaluation.
#' @param cv_initial    Min training obs per CV fold (NULL = auto).
#' @param cv_skip       Obs skipped between fold origins.
#' @param cv_cumulative Expanding (TRUE) or sliding (FALSE) window.
#' @param xreg          Optional external regressors matrix (train).
#' @param xreg_future   External regressors for forecast horizon.
#' @param seed          RNG seed.
#' @param verbose       0 = silent, 1 = progress.
#'
#' @return Object of class "auto_arima_ts".
#' @export
auto_arima_ts <- function(data,
                          date_col      = "date",
                          value_col     = "value",
                          horizon       = 12L,
                          level         = c(80, 95),
                          stepwise      = TRUE,
                          approximation = NULL,
                          lambda        = NULL,
                          run_cv        = TRUE,
                          cv_initial    = NULL,
                          cv_skip       = 0L,
                          cv_cumulative = TRUE,
                          xreg          = NULL,
                          xreg_future   = NULL,
                          seed          = 42L,
                          verbose       = 1L) {
  set.seed(seed)
  stopifnot(is.data.frame(data),
            date_col  %in% names(data),
            value_col %in% names(data))

  df <- data %>%
    rename(date = !!sym(date_col), value = !!sym(value_col)) %>%
    arrange(date) %>% select(date, value)

  if (anyNA(df$value)) {
    warning("Missing values detected — interpolating.")
    df$value <- zoo::na.approx(df$value, na.rm = FALSE)
  }

  ts_obj <- .df_to_ts(df)
  freq   <- .ts_freq(df$date)

  if (verbose > 0L) cat("\n── Auto-ARIMA: fitting model ────────────────────────────────\n")

  # ── Fit auto.arima ──────────────────────────────────────────────
  model <- forecast::auto.arima(
    y             = ts_obj,
    xreg          = xreg,
    stepwise      = stepwise,
    approximation = approximation,
    lambda        = lambda,
    ic            = "aicc",
    trace         = verbose > 0L
  )

  if (verbose > 0L) {
    cat(sprintf("\n  Selected order: ARIMA(%d,%d,%d)",
                model$arma[1], model$arma[6], model$arma[2]))
    if (freq > 1L)
      cat(sprintf(" × (%d,%d,%d)[%d]",
                  model$arma[3], model$arma[7], model$arma[4], freq))
    cat("\n")
  }

  # ── Forecast ────────────────────────────────────────────────────
  fc_obj <- forecast::forecast(model, h = horizon,
                               level = level, xreg = xreg_future)
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by = freq_str,
                   length.out = horizon + 1L)[-1L]

  forecast_df <- tibble(
    date       = fut_dates,
    forecast   = as.numeric(fc_obj$mean),
    lower_80   = as.numeric(fc_obj$lower[, 1]),
    upper_80   = as.numeric(fc_obj$upper[, 1]),
    lower_95   = if (length(level) >= 2L) as.numeric(fc_obj$lower[, 2])
                 else NA_real_,
    upper_95   = if (length(level) >= 2L) as.numeric(fc_obj$upper[, 2])
                 else NA_real_
  )

  # ── Rolling-origin CV ───────────────────────────────────────────
  cv_results <- NULL
  if (run_cv) {
    if (is.null(cv_initial))
      cv_initial <- max(as.integer(nrow(df) * 0.5),
                        2L * freq + horizon)

    if (verbose > 0L)
      cat(sprintf("\n── Auto-ARIMA: rolling_origin CV (initial=%d) ───────────────\n",
                  cv_initial))

    splits   <- rolling_origin(df, initial = cv_initial,
                               assess = horizon, skip = cv_skip,
                               cumulative = cv_cumulative)
    fold_res <- vector("list", nrow(splits))

    for (i in seq_len(nrow(splits))) {
      if (verbose > 0L) cat(sprintf("  Fold %d / %d\r", i, nrow(splits)))
      tr  <- analysis(splits$splits[[i]])
      ts_i <- ts(tr$value,
                 frequency = freq,
                 start     = c(as.integer(format(min(tr$date), "%Y")),
                               if (freq == 12L)
                                 as.integer(format(min(tr$date), "%m"))
                               else 1L))
      m_i  <- tryCatch(
        forecast::auto.arima(ts_i, stepwise = stepwise,
                             approximation = approximation,
                             lambda = lambda, ic = "aicc"),
        error = function(e) NULL
      )
      av   <- assessment(splits$splits[[i]])$value
      av   <- av[seq_len(min(horizon, length(av)))]

      if (!is.null(m_i)) {
        fc_i <- as.numeric(forecast::forecast(m_i, h = horizon)$mean)
        fc_i <- fc_i[seq_along(av)]
        m    <- .safe_metrics(av, fc_i)
        fold_res[[i]] <- list(
          fold      = i, n_train = nrow(tr),
          actual    = av, predicted = fc_i,
          RMSE = m$RMSE, MAE = m$MAE, MAPE = m$MAPE,
          SMAPE = m$SMAPE, MBE = m$MBE
        )
      }
    }
    if (verbose > 0L) cat("\n")

    fold_res <- Filter(Negate(is.null), fold_res)
    cv_df <- dplyr::bind_rows(lapply(fold_res, function(r)
      tibble(fold = r$fold, n_train = r$n_train,
             RMSE = r$RMSE, MAE = r$MAE,
             MAPE = r$MAPE, SMAPE = r$SMAPE, MBE = r$MBE)))

    cv_summary <- cv_df %>%
      summarise(across(c(RMSE,MAE,MAPE,SMAPE,MBE),
                       list(mean = mean, sd = sd), na.rm = TRUE,
                       .names = "{.col}__{.fn}")) %>%
      pivot_longer(everything(),
                   names_to = c("metric","stat"), names_sep = "__") %>%
      pivot_wider(names_from = stat, values_from = value)

    cv_results <- list(cv_metrics    = cv_df,
                       summary       = cv_summary,
                       fold_forecasts = fold_res)

    if (verbose > 0L) {
      cat("\n── Auto-ARIMA: CV Summary ────────────────────────────────────\n")
      print(cv_summary)
    }
  }

  structure(
    list(model      = model,
         forecast   = forecast_df,
         cv_results = cv_results,
         data       = df,
         horizon    = horizon,
         level      = level,
         ts_obj     = ts_obj,
         freq       = freq),
    class = "auto_arima_ts"
  )
}

# ── print.auto_arima_ts ─────────────────────────────────────────
#' @export
print.auto_arima_ts <- function(x, digits = 4L, ...) {
  m <- x$model
  cat("\n", .box("═"), "\n", sep = "")
  cat("  Auto-ARIMA  (forecast::auto.arima)\n")
  cat(.box("═"), "\n")
  cat(.hdr("Model"), "\n")
  cat(sprintf("  Order          : ARIMA(%d,%d,%d)\n",
              m$arma[1], m$arma[6], m$arma[2]))
  if (x$freq > 1L)
    cat(sprintf("  Seasonal       : (%d,%d,%d)[%d]\n",
                m$arma[3], m$arma[7], m$arma[4], x$freq))
  cat(sprintf("  AICc           : %.4f\n", m$aicc))
  cat(sprintf("  Log-lik        : %.4f\n", m$loglik))
  cat(.hdr("Data"), "\n")
  cat(sprintf("  n              : %d\n", nrow(x$data)))
  cat(sprintf("  Date range     : %s → %s\n",
              format(min(x$data$date)), format(max(x$data$date))))
  cat(.hdr("Forecast"), "\n")
  cat(sprintf("  Horizon        : %d steps\n", x$horizon))
  fc <- head(x$forecast, 4L)
  cat(sprintf("  %-12s %12s %12s %12s\n",
              "Date","Forecast","Lower 95%","Upper 95%"))
  for (i in seq_len(nrow(fc)))
    cat(sprintf("  %-12s %12s %12s %12s\n",
                format(fc$date[i]),
                .fmt(fc$forecast[i],  digits),
                .fmt(fc$lower_95[i],  digits),
                .fmt(fc$upper_95[i],  digits)))
  if (nrow(x$forecast) > 4L)
    cat(sprintf("  … %d more rows\n", nrow(x$forecast) - 4L))
  if (!is.null(x$cv_results)) {
    cat(.hdr("CV Summary"), "\n")
    print(x$cv_results$summary, n = Inf)
  }
  cat(.box("═"), "\n\n")
  invisible(x)
}

# ── summary.auto_arima_ts ───────────────────────────────────────
#' @export
summary.auto_arima_ts <- function(object, ...) {
  cat("\n── Full ARIMA Model Summary ─────────────────────────────────\n")
  print(summary(object$model))
  if (!is.null(object$cv_results)) {
    cat("\n── Rolling-Origin CV Aggregate ──────────────────────────────\n")
    print(object$cv_results$summary)
  }
  invisible(object)
}

# ── predict.auto_arima_ts ───────────────────────────────────────
#' @export
predict.auto_arima_ts <- function(object,
                                  horizon  = NULL,
                                  newdata  = NULL,
                                  level    = NULL,
                                  xreg     = NULL,
                                  ...) {
  h     <- horizon %||% object$horizon
  lv    <- level   %||% object$level
  model <- if (!is.null(newdata)) {
    ts_new <- .df_to_ts(newdata %>%
                          rename(date = 1, value = 2))
    tryCatch(
      forecast::Arima(ts_new, model = object$model),
      error = function(e) object$model
    )
  } else object$model

  fc <- forecast::forecast(model, h = h, level = lv, xreg = xreg)
  freq_str <- infer_frequency(object$data$date)
  base_date <- if (!is.null(newdata)) max(newdata[[1]])
               else max(object$data$date)
  fut_dates <- seq(base_date, by = freq_str, length.out = h + 1L)[-1L]

  tibble(
    date      = fut_dates,
    predicted = as.numeric(fc$mean),
    lower_95  = as.numeric(fc$lower[, ncol(fc$lower)]),
    upper_95  = as.numeric(fc$upper[, ncol(fc$upper)])
  )
}

# ── fitted.auto_arima_ts ────────────────────────────────────────
#' @export
fitted.auto_arima_ts <- function(object, ...) {
  fv <- as.numeric(fitted(object$model))
  tibble(
    date     = object$data$date[seq_along(fv)],
    fitted   = fv,
    actual   = object$data$value[seq_along(fv)],
    residual = object$data$value[seq_along(fv)] - fv
  )
}

# ── residuals.auto_arima_ts ─────────────────────────────────────
#' @export
residuals.auto_arima_ts <- function(object, ...) {
  as.numeric(residuals(object$model))
}

# ── predicted.auto_arima_ts  (OOS) ─────────────────────────────
#' @export
predicted.auto_arima_ts <- function(object,
                                    aggregate = FALSE, ...) {
  if (is.null(object$cv_results))
    stop("No CV results. Re-run with run_cv = TRUE.")

  rows <- lapply(object$cv_results$fold_forecasts, function(r) {
    n <- length(r$actual)
    tibble(fold      = r$fold,
           h         = seq_len(n),
           actual    = r$actual,
           predicted = r$predicted,
           residual  = r$actual - r$predicted)
  })
  oos <- dplyr::bind_rows(rows)

  if (!aggregate) return(oos)

  oos %>%
    group_by(h) %>%
    summarise(mean_actual    = mean(actual),
              mean_predicted = mean(predicted),
              mean_residual  = mean(residual),
              sd_residual    = sd(residual),
              RMSE = sqrt(mean(residual^2)),
              MAE  = mean(abs(residual)),
              .groups = "drop")
}

# ── cv_performance.auto_arima_ts ────────────────────────────────
#' @export
cv_performance.auto_arima_ts <- function(object, ...) {
  if (is.null(object$cv_results))
    stop("No CV results. Re-run with run_cv = TRUE.")

  pf <- object$cv_results$cv_metrics
  cat("\n── Auto-ARIMA OOS CV Performance ───────────────────────────\n")
  print(object$cv_results$summary)
  invisible(list(per_fold  = pf,
                 aggregate = object$cv_results$summary,
                 raw_oos   = predicted(object)))
}

# ── plot.auto_arima_ts ──────────────────────────────────────────
#' @export
plot.auto_arima_ts <- function(x, last_n = NULL, which = NULL, ...) {
  btheme <- theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11))
  panels <- list()

  # P1: Forecast
  hist_df <- x$data
  if (!is.null(last_n)) hist_df <- tail(hist_df, last_n)
  fc  <- x$forecast

  panels$forecast <- ggplot() +
    geom_line(data = hist_df, aes(date, value),
              colour = .PAL$actual, linewidth = 0.8) +
    geom_ribbon(data = fc,
                aes(date, ymin = lower_95, ymax = upper_95),
                fill = .PAL$arima, alpha = 0.15) +
    geom_ribbon(data = fc,
                aes(date, ymin = lower_80, ymax = upper_80),
                fill = .PAL$arima, alpha = 0.20) +
    geom_line(data = fc, aes(date, forecast),
              colour = .PAL$arima, linewidth = 1.0, linetype = "dashed") +
    geom_point(data = fc, aes(date, forecast),
               colour = .PAL$arima, size = 2.0,
               shape = 21, fill = "white", stroke = 1.2) +
    labs(title    = sprintf("Auto-ARIMA(%d,%d,%d) Forecast",
                            x$model$arma[1], x$model$arma[6],
                            x$model$arma[2]),
         subtitle = "Shaded = 80% and 95% PI",
         x = NULL, y = "Value") + btheme

  # P2: Residuals
  res_df <- fitted(x)
  panels$residuals <- ggplot(res_df, aes(date, residual)) +
    geom_hline(yintercept = 0, colour = .PAL$neutral, linewidth = 0.6) +
    geom_line(colour = .PAL$arima, linewidth = 0.7, alpha = 0.7) +
    geom_point(colour = .PAL$arima, size = 1.0, alpha = 0.5) +
    labs(title = "ARIMA In-Sample Residuals", x = NULL, y = "Residual") +
    btheme

  # P3: Residual ACF
  res_vec <- residuals(x)
  n_lag   <- min(30L, floor(length(res_vec) / 2L))
  acf_obj <- acf(res_vec, lag.max = n_lag, plot = FALSE)
  ci_line <- qnorm(0.975) / sqrt(length(res_vec))
  acf_df  <- tibble(lag = as.numeric(acf_obj$lag[-1]),
                    acf = as.numeric(acf_obj$acf[-1]))
  panels$acf <- ggplot(acf_df, aes(lag, acf)) +
    geom_hline(yintercept = 0, colour = "grey30", linewidth = 0.5) +
    geom_hline(yintercept = c(-ci_line, ci_line),
               linetype = "dashed", colour = .PAL$direct, linewidth = 0.6) +
    geom_segment(aes(xend = lag, yend = 0,
                     colour = abs(acf) > ci_line), linewidth = 1.0) +
    geom_point(aes(colour = abs(acf) > ci_line), size = 1.8) +
    scale_colour_manual(values = c("FALSE" = .PAL$arima,
                                   "TRUE"  = .PAL$direct),
                        guide = "none") +
    labs(title = "Residual ACF", x = "Lag", y = "ACF") + btheme

  # P4: CV per fold (if available)
  if (!is.null(x$cv_results)) {
    panels$cv <- ggplot(x$cv_results$cv_metrics,
                        aes(fold, RMSE)) +
      geom_line(colour = .PAL$arima, linewidth = 0.9) +
      geom_point(colour = .PAL$arima, size = 2.0,
                 shape = 21, fill = "white", stroke = 1.2) +
      geom_hline(
        yintercept = mean(x$cv_results$cv_metrics$RMSE, na.rm = TRUE),
        linetype = "dashed", colour = .PAL$direct, linewidth = 0.6) +
      labs(title = "CV RMSE per Fold",
           subtitle = "Dashed = mean across folds",
           x = "Fold", y = "RMSE") + btheme
  }

  avail <- if (is.null(which)) seq_along(panels) else which
  pw    <- patchwork::wrap_plots(panels[avail], ncol = 2L) +
    patchwork::plot_annotation(
      title = "Auto-ARIMA Diagnostic Dashboard",
      theme = theme(plot.title = element_text(
        face = "bold", size = 14, hjust = 0.5))
    )
  print(pw)
  invisible(pw)
}


# ================================================================
# PART F — AUTO-ARFIMA MODEL
# ================================================================

#' Fit Auto-ARFIMA with full pipeline (long-memory time series)
#'
#' Wraps fracdiff::fracdiff() + forecast::arfima() with:
#'   • Automatic fractional differencing order (d) estimation
#'   • Stationarity diagnostics (ADF + KPSS)
#'   • rolling_origin OOS cross-validation
#'   • Unified S3 class "auto_arfima_ts"
#'
#' @param data          Data frame with `date` + `value` columns.
#' @param date_col      Date column name.
#' @param value_col     Value column name.
#' @param horizon       Forecast steps ahead.
#' @param level         Prediction interval level(s).
#' @param drange        Range for d search (default c(0, 0.5)).
#' @param ar_max        Max AR order for conditional mean model.
#' @param ma_max        Max MA order for conditional mean model.
#' @param run_cv        Run rolling_origin OOS evaluation.
#' @param cv_initial    Min training obs per CV fold (NULL = auto).
#' @param cv_skip       Obs skipped between fold origins.
#' @param cv_cumulative Expanding (TRUE) or sliding (FALSE) window.
#' @param seed          RNG seed.
#' @param verbose       0 = silent, 1 = progress.
#'
#' @return Object of class "auto_arfima_ts".
#' @export
auto_arfima_ts <- function(data,
                           date_col      = "date",
                           value_col     = "value",
                           horizon       = 12L,
                           level         = c(80, 95),
                           drange        = c(0, 0.5),
                           ar_max        = 5L,
                           ma_max        = 5L,
                           run_cv        = TRUE,
                           cv_initial    = NULL,
                           cv_skip       = 0L,
                           cv_cumulative = TRUE,
                           seed          = 42L,
                           verbose       = 1L) {
  set.seed(seed)
  stopifnot(is.data.frame(data),
            date_col  %in% names(data),
            value_col %in% names(data))

  df <- data %>%
    rename(date = !!sym(date_col), value = !!sym(value_col)) %>%
    arrange(date) %>% select(date, value)

  if (anyNA(df$value)) {
    warning("Missing values detected — interpolating.")
    df$value <- zoo::na.approx(df$value, na.rm = FALSE)
  }

  ts_obj <- .df_to_ts(df)
  freq   <- .ts_freq(df$date)

  # ── Stationarity diagnostics ────────────────────────────────────
  stationarity <- list()
  tryCatch({
    adf  <- tseries::adf.test(ts_obj)
    kpss <- tseries::kpss.test(ts_obj)
    stationarity <- list(
      adf_stat  = as.numeric(adf$statistic),
      adf_pval  = adf$p.value,
      kpss_stat = as.numeric(kpss$statistic),
      kpss_pval = kpss$p.value,
      is_stationary = adf$p.value < 0.05 && kpss$p.value > 0.05
    )
    if (verbose > 0L) {
      cat(sprintf("\n── Stationarity Tests ───────────────────────────────────────\n"))
      cat(sprintf("  ADF  p-value : %.4f  %s\n",
                  adf$p.value,
                  if (adf$p.value < 0.05) "(stationary)" else "(non-stationary)"))
      cat(sprintf("  KPSS p-value : %.4f  %s\n",
                  kpss$p.value,
                  if (kpss$p.value > 0.05) "(stationary)" else "(non-stationary)"))
    }
  }, error = function(e) NULL)

  # ── Estimate fractional d via fracdiff ──────────────────────────
  if (verbose > 0L)
    cat("\n── Auto-ARFIMA: estimating fractional d ─────────────────────\n")

  fd_fit <- fracdiff::fracdiff(ts_obj, drange = drange,
                               ar = ar_max, ma = ma_max)
  d_hat  <- fd_fit$d

  if (verbose > 0L)
    cat(sprintf("  Estimated d  : %.6f\n", d_hat))

  # ── Fit ARFIMA via forecast::arfima ─────────────────────────────
  if (verbose > 0L)
    cat("\n── Auto-ARFIMA: fitting ARIMA on fractionally differenced series\n")

  model <- tryCatch(
    forecast::arfima(ts_obj, drange = drange, estim = "mle"),
    error = function(e) {
      warning("arfima() failed; falling back to fracdiff model.")
      fd_fit
    }
  )

  # ── Forecast ────────────────────────────────────────────────────
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by = freq_str,
                   length.out = horizon + 1L)[-1L]

  fc_obj <- tryCatch(
    forecast::forecast(model, h = horizon, level = level),
    error = function(e) NULL
  )

  forecast_df <- if (!is.null(fc_obj)) {
    tibble(
      date     = fut_dates,
      forecast = as.numeric(fc_obj$mean),
      lower_80 = as.numeric(fc_obj$lower[, 1]),
      upper_80 = as.numeric(fc_obj$upper[, 1]),
      lower_95 = if (length(level) >= 2L)
                   as.numeric(fc_obj$lower[, 2]) else NA_real_,
      upper_95 = if (length(level) >= 2L)
                   as.numeric(fc_obj$upper[, 2]) else NA_real_
    )
  } else {
    # Manual bootstrap forecast from fracdiff
    fd_diff  <- fracdiff::diffseries(as.numeric(ts_obj), d_hat)
    ar_model <- forecast::auto.arima(ts(fd_diff, frequency = freq),
                                     max.p = ar_max, max.q = ma_max,
                                     d = 0L, stepwise = TRUE)
    preds_diff <- as.numeric(forecast::forecast(ar_model, h = horizon)$mean)
    preds_raw  <- fracdiff::diffseries(
      c(as.numeric(ts_obj), preds_diff), -d_hat)
    preds_fc   <- tail(preds_raw, horizon)
    tibble(
      date     = fut_dates,
      forecast = preds_fc,
      lower_80 = preds_fc - qnorm(0.90) * sd(residuals(ar_model)),
      upper_80 = preds_fc + qnorm(0.90) * sd(residuals(ar_model)),
      lower_95 = preds_fc - qnorm(0.975) * sd(residuals(ar_model)),
      upper_95 = preds_fc + qnorm(0.975) * sd(residuals(ar_model))
    )
  }

  # ── Rolling-origin CV ───────────────────────────────────────────
  cv_results <- NULL
  if (run_cv) {
    if (is.null(cv_initial))
      cv_initial <- max(as.integer(nrow(df) * 0.5),
                        2L * freq + horizon)

    if (verbose > 0L)
      cat(sprintf("\n── Auto-ARFIMA: rolling_origin CV (initial=%d) ──────────────\n",
                  cv_initial))

    splits   <- rolling_origin(df, initial = cv_initial,
                               assess = horizon, skip = cv_skip,
                               cumulative = cv_cumulative)
    fold_res <- vector("list", nrow(splits))

    for (i in seq_len(nrow(splits))) {
      if (verbose > 0L) cat(sprintf("  Fold %d / %d\r", i, nrow(splits)))
      tr   <- analysis(splits$splits[[i]])
      ts_i <- ts(tr$value, frequency = freq,
                 start = c(as.integer(format(min(tr$date), "%Y")),
                           if (freq == 12L)
                             as.integer(format(min(tr$date), "%m"))
                           else 1L))
      m_i  <- tryCatch(
        forecast::arfima(ts_i, drange = drange, estim = "mle"),
        error = function(e) NULL
      )
      av <- assessment(splits$splits[[i]])$value
      av <- av[seq_len(min(horizon, length(av)))]

      if (!is.null(m_i)) {
        fc_i <- tryCatch(
          as.numeric(forecast::forecast(m_i, h = horizon)$mean),
          error = function(e) rep(NA_real_, horizon)
        )
        fc_i <- fc_i[seq_along(av)]
        m    <- .safe_metrics(av, fc_i)
        fold_res[[i]] <- list(
          fold = i, n_train = nrow(tr),
          actual = av, predicted = fc_i,
          RMSE = m$RMSE, MAE = m$MAE, MAPE = m$MAPE,
          SMAPE = m$SMAPE, MBE = m$MBE,
          d_hat = tryCatch(fracdiff::fracdiff(ts_i, drange = drange)$d,
                           error = function(e) NA_real_)
        )
      }
    }
    if (verbose > 0L) cat("\n")

    fold_res <- Filter(Negate(is.null), fold_res)
    cv_df <- dplyr::bind_rows(lapply(fold_res, function(r)
      tibble(fold = r$fold, n_train = r$n_train,
             RMSE = r$RMSE, MAE = r$MAE,
             MAPE = r$MAPE, SMAPE = r$SMAPE,
             MBE = r$MBE, d_hat = r$d_hat)))

    cv_summary <- cv_df %>%
      summarise(across(c(RMSE,MAE,MAPE,SMAPE,MBE,d_hat),
                       list(mean = mean, sd = sd), na.rm = TRUE,
                       .names = "{.col}__{.fn}")) %>%
      pivot_longer(everything(),
                   names_to  = c("metric","stat"),
                   names_sep = "__") %>%
      pivot_wider(names_from = stat, values_from = value)

    cv_results <- list(cv_metrics     = cv_df,
                       summary        = cv_summary,
                       fold_forecasts = fold_res)

    if (verbose > 0L) {
      cat("\n── Auto-ARFIMA: CV Summary ───────────────────────────────────\n")
      print(cv_summary)
    }
  }

  structure(
    list(model         = model,
         fd_fit        = fd_fit,
         d_hat         = d_hat,
         forecast      = forecast_df,
         cv_results    = cv_results,
         stationarity  = stationarity,
         data          = df,
         horizon       = horizon,
         level         = level,
         ts_obj        = ts_obj,
         freq          = freq),
    class = "auto_arfima_ts"
  )
}

# ── print.auto_arfima_ts ────────────────────────────────────────
#' @export
print.auto_arfima_ts <- function(x, digits = 4L, ...) {
  cat("\n", .box("═"), "\n", sep = "")
  cat("  Auto-ARFIMA  (fracdiff + forecast::arfima)\n")
  cat(.box("═"), "\n")
  cat(.hdr("Long-Memory Model"), "\n")
  cat(sprintf("  Fractional d   : %.6f\n", x$d_hat))
  cat(sprintf("  Interpretation : %s\n",
              dplyr::case_when(
                x$d_hat < 0.1  ~ "Short memory (d ≈ 0)",
                x$d_hat < 0.5  ~ "Long memory: stationary (0 < d < 0.5)",
                x$d_hat < 1.0  ~ "Long memory: non-stationary (0.5 ≤ d < 1)",
                TRUE            ~ "Integer integration (d ≥ 1)"
              )))
  if (!is.null(x$stationarity) && length(x$stationarity) > 0L) {
    cat(.hdr("Stationarity"), "\n")
    cat(sprintf("  ADF p-value    : %.4f  %s\n",
                x$stationarity$adf_pval,
                if (!is.na(x$stationarity$adf_pval) &&
                    x$stationarity$adf_pval < 0.05) "✓ stationary"
                else "⚠ non-stationary"))
    cat(sprintf("  KPSS p-value   : %.4f  %s\n",
                x$stationarity$kpss_pval,
                if (!is.na(x$stationarity$kpss_pval) &&
                    x$stationarity$kpss_pval > 0.05) "✓ stationary"
                else "⚠ non-stationary"))
  }
  cat(.hdr("Data"), "\n")
  cat(sprintf("  n              : %d\n", nrow(x$data)))
  cat(sprintf("  Date range     : %s → %s\n",
              format(min(x$data$date)), format(max(x$data$date))))
  cat(.hdr("Forecast"), "\n")
  cat(sprintf("  Horizon        : %d steps\n", x$horizon))
  fc <- head(x$forecast, 4L)
  cat(sprintf("  %-12s %12s %12s %12s\n",
              "Date","Forecast","Lower 95%","Upper 95%"))
  for (i in seq_len(nrow(fc)))
    cat(sprintf("  %-12s %12s %12s %12s\n",
                format(fc$date[i]),
                .fmt(fc$forecast[i], digits),
                .fmt(fc$lower_95[i], digits),
                .fmt(fc$upper_95[i], digits)))
  if (!is.null(x$cv_results)) {
    cat(.hdr("CV Summary"), "\n")
    print(x$cv_results$summary, n = Inf)
  }
  cat(.box("═"), "\n\n")
  invisible(x)
}

# ── summary.auto_arfima_ts ──────────────────────────────────────
#' @export
summary.auto_arfima_ts <- function(object, ...) {
  cat("\n── ARFIMA Model Summary ─────────────────────────────────────\n")
  cat(sprintf("  Fractional differencing d = %.6f\n", object$d_hat))
  tryCatch(print(summary(object$model)), error = function(e)
    print(summary(object$fd_fit)))
  if (!is.null(object$cv_results)) {
    cat("\n── Rolling-Origin CV Aggregate ──────────────────────────────\n")
    print(object$cv_results$summary)
  }
  invisible(object)
}

# ── predict.auto_arfima_ts ──────────────────────────────────────
#' @export
predict.auto_arfima_ts <- function(object,
                                   horizon = NULL,
                                   level   = NULL, ...) {
  h  <- horizon %||% object$horizon
  lv <- level   %||% object$level
  fc <- tryCatch(
    forecast::forecast(object$model, h = h, level = lv),
    error = function(e) NULL
  )
  if (is.null(fc)) {
    warning("Could not generate forecast from stored model.")
    return(object$forecast[seq_len(h), ])
  }
  freq_str  <- infer_frequency(object$data$date)
  fut_dates <- seq(max(object$data$date), by = freq_str,
                   length.out = h + 1L)[-1L]
  tibble(
    date      = fut_dates,
    predicted = as.numeric(fc$mean),
    lower_95  = as.numeric(fc$lower[, ncol(fc$lower)]),
    upper_95  = as.numeric(fc$upper[, ncol(fc$upper)])
  )
}

# ── fitted.auto_arfima_ts ───────────────────────────────────────
#' @export
fitted.auto_arfima_ts <- function(object, ...) {
  fv <- tryCatch(as.numeric(fitted(object$model)),
                 error = function(e) rep(NA_real_, nrow(object$data)))
  n  <- min(length(fv), nrow(object$data))
  tibble(
    date     = object$data$date[seq_len(n)],
    fitted   = fv[seq_len(n)],
    actual   = object$data$value[seq_len(n)],
    residual = object$data$value[seq_len(n)] - fv[seq_len(n)]
  )
}

# ── residuals.auto_arfima_ts ────────────────────────────────────
#' @export
residuals.auto_arfima_ts <- function(object, ...) {
  tryCatch(as.numeric(residuals(object$model)),
           error = function(e) as.numeric(residuals(object$fd_fit)))
}

# ── predicted.auto_arfima_ts  (OOS) ────────────────────────────
#' @export
predicted.auto_arfima_ts <- function(object,
                                     aggregate = FALSE, ...) {
  if (is.null(object$cv_results))
    stop("No CV results. Re-run with run_cv = TRUE.")

  rows <- lapply(object$cv_results$fold_forecasts, function(r) {
    n <- length(r$actual)
    tibble(fold      = r$fold,
           h         = seq_len(n),
           actual    = r$actual,
           predicted = r$predicted,
           residual  = r$actual - r$predicted,
           d_hat     = r$d_hat)
  })
  oos <- dplyr::bind_rows(rows)
  if (!aggregate) return(oos)

  oos %>%
    group_by(h) %>%
    summarise(mean_actual    = mean(actual),
              mean_predicted = mean(predicted),
              mean_residual  = mean(residual),
              sd_residual    = sd(residual),
              mean_d         = mean(d_hat, na.rm = TRUE),
              RMSE = sqrt(mean(residual^2)),
              MAE  = mean(abs(residual)),
              .groups = "drop")
}

# ── cv_performance.auto_arfima_ts ───────────────────────────────
#' @export
cv_performance.auto_arfima_ts <- function(object, ...) {
  if (is.null(object$cv_results))
    stop("No CV results. Re-run with run_cv = TRUE.")
  cat("\n── Auto-ARFIMA OOS CV Performance ──────────────────────────\n")
  print(object$cv_results$summary)
  invisible(list(per_fold  = object$cv_results$cv_metrics,
                 aggregate = object$cv_results$summary,
                 raw_oos   = predicted(object)))
}

# ── plot.auto_arfima_ts ─────────────────────────────────────────
#' @export
plot.auto_arfima_ts <- function(x, last_n = NULL, ...) {
  btheme <- theme_minimal(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11))
  panels <- list()

  # P1: Forecast
  hist_df <- x$data
  if (!is.null(last_n)) hist_df <- tail(hist_df, last_n)
  fc <- x$forecast

  panels$forecast <- ggplot() +
    geom_line(data = hist_df, aes(date, value),
              colour = .PAL$actual, linewidth = 0.8) +
    geom_ribbon(data = fc,
                aes(date, ymin = lower_95, ymax = upper_95),
                fill = .PAL$arfima, alpha = 0.15) +
    geom_ribbon(data = fc,
                aes(date, ymin = lower_80, ymax = upper_80),
                fill = .PAL$arfima, alpha = 0.20) +
    geom_line(data = fc, aes(date, forecast),
              colour = .PAL$arfima, linewidth = 1.0, linetype = "dashed") +
    geom_point(data = fc, aes(date, forecast),
               colour = .PAL$arfima, size = 2.0,
               shape = 21, fill = "white", stroke = 1.2) +
    labs(title    = sprintf("Auto-ARFIMA Forecast  (d = %.4f)", x$d_hat),
         subtitle = "Shaded = 80% and 95% PI",
         x = NULL, y = "Value") + btheme

  # P2: d-hat per fold
  if (!is.null(x$cv_results) &&
      "d_hat" %in% names(x$cv_results$cv_metrics)) {
    panels$d_hat <- ggplot(x$cv_results$cv_metrics,
                           aes(fold, d_hat)) +
      geom_line(colour = .PAL$arfima, linewidth = 0.9) +
      geom_point(colour = .PAL$arfima, size = 2.0,
                 shape = 21, fill = "white", stroke = 1.2) +
      geom_hline(yintercept = x$d_hat,
                 linetype = "dashed", colour = .PAL$direct) +
      labs(title    = "Fractional d per CV Fold",
           subtitle = sprintf("Full-sample d = %.4f (dashed)", x$d_hat),
           x = "Fold", y = "d") + btheme
  }

  # P3: Residuals
  res_df <- fitted(x)
  panels$residuals <- ggplot(filter(res_df, !is.na(residual)),
                             aes(date, residual)) +
    geom_hline(yintercept = 0, colour = .PAL$neutral, linewidth = 0.6) +
    geom_line(colour = .PAL$arfima, linewidth = 0.7, alpha = 0.7) +
    labs(title = "ARFIMA In-Sample Residuals", x = NULL, y = "Residual") +
    btheme

  # P4: CV RMSE
  if (!is.null(x$cv_results)) {
    panels$cv_rmse <- ggplot(x$cv_results$cv_metrics,
                             aes(fold, RMSE)) +
      geom_line(colour = .PAL$arfima, linewidth = 0.9) +
      geom_point(colour = .PAL$arfima, size = 2.0,
                 shape = 21, fill = "white", stroke = 1.2) +
      geom_hline(
        yintercept = mean(x$cv_results$cv_metrics$RMSE, na.rm = TRUE),
        linetype = "dashed", colour = .PAL$direct, linewidth = 0.6) +
      labs(title = "CV RMSE per Fold", x = "Fold", y = "RMSE") + btheme
  }

  pw <- patchwork::wrap_plots(panels, ncol = 2L) +
    patchwork::plot_annotation(
      title = "Auto-ARFIMA Diagnostic Dashboard",
      theme = theme(plot.title = element_text(
        face = "bold", size = 14, hjust = 0.5))
    )
  print(pw)
  invisible(pw)
}


# ================================================================
# PART G — UNIFIED MODEL COMPARISON
# ================================================================

#' Compare RNN models + ARIMA + ARFIMA on the same dataset
#'
#' Trains all requested model families, extracts CV performance,
#' and returns a unified ranking table and per-model results.
#'
#' @param data          Data frame with date + value columns.
#' @param rnn_models    Character vector: "lstm","gru","rnn" (or NULL to skip).
#' @param run_arima     Fit Auto-ARIMA.
#' @param run_arfima    Fit Auto-ARFIMA.
#' @param horizon       Forecast horizon (shared by all models).
#' @param rank_by       Metric for ranking: "RMSE" | "MAE" | "MAPE".
#' @param rnn_strategy  Strategy for RNN CV ranking: "recursive" | "direct".
#' @param cv_initial    Min training obs for CV (NULL = auto).
#' @param ...           Additional args forwarded to auto_rnn_torch().
#'
#' @return list(ranking, results, plot)
#' @export
compare_all_models <- function(data,
                               rnn_models   = c("lstm","gru","rnn"),
                               run_arima    = TRUE,
                               run_arfima   = TRUE,
                               horizon      = 12L,
                               rank_by      = "RMSE",
                               rnn_strategy = "recursive",
                               cv_initial   = NULL,
                               ...) {
  results <- list()

  # ── RNN models ─────────────────────────────────────────────────
  if (!is.null(rnn_models) && length(rnn_models) > 0L) {
    for (mt in rnn_models) {
      cat(sprintf("\n%s\n  Training Auto%s\n%s\n",
                  strrep("═", 50), toupper(mt), strrep("═", 50)))
      results[[toupper(mt)]] <- auto_rnn_torch(
        data       = data,
        model_type = mt,
        horizon    = horizon,
        run_cv     = TRUE,
        cv_initial = cv_initial,
        ...
      )
    }
  }

  # ── Auto-ARIMA ─────────────────────────────────────────────────
  if (run_arima) {
    cat(sprintf("\n%s\n  Training Auto-ARIMA\n%s\n",
                strrep("═", 50), strrep("═", 50)))
    results[["ARIMA"]] <- auto_arima_ts(
      data       = data,
      horizon    = horizon,
      run_cv     = TRUE,
      cv_initial = cv_initial
    )
  }

  # ── Auto-ARFIMA ────────────────────────────────────────────────
  if (run_arfima) {
    cat(sprintf("\n%s\n  Training Auto-ARFIMA\n%s\n",
                strrep("═", 50), strrep("═", 50)))
    results[["ARFIMA"]] <- auto_arfima_ts(
      data       = data,
      horizon    = horizon,
      run_cv     = TRUE,
      cv_initial = cv_initial
    )
  }

  # ── Extract CV metrics for ranking ─────────────────────────────
  .pull_metric <- function(name, res, metric) {
    if (inherits(res, "auto_rnn")) {
      smry <- res$cv_results$summary
      if (is.null(smry)) return(NA_real_)
      key  <- paste0(metric, "_", rnn_strategy)
      row  <- smry[smry$metric == key, "mean", drop = TRUE]
      if (length(row) == 0L) NA_real_ else row[[1L]]
    } else if (inherits(res, c("auto_arima_ts","auto_arfima_ts"))) {
      smry <- res$cv_results$summary
      if (is.null(smry)) return(NA_real_)
      row  <- smry[smry$metric == metric, "mean", drop = TRUE]
      if (length(row) == 0L) NA_real_ else row[[1L]]
    } else NA_real_
  }

  metrics_to_show <- c("RMSE","MAE","MAPE","SMAPE")
  ranking_rows <- lapply(names(results), function(nm) {
    res <- results[[nm]]
    row <- tibble(model = nm)
    for (m in metrics_to_show)
      row[[m]] <- .pull_metric(nm, res, m)
    row
  })

  ranking <- dplyr::bind_rows(ranking_rows) %>%
    arrange(.data[[rank_by]])

  # ── Print comparison table ──────────────────────────────────────
  cat("\n", strrep("═", 62), "\n", sep = "")
  cat("  Unified Model Comparison — OOS CV Performance\n")
  cat(strrep("═", 62), "\n", sep = "")
  cat(sprintf("  Ranked by: %s  |  %s\n\n", rank_by,
              if (!is.null(rnn_models))
                sprintf("RNN strategy: %s", rnn_strategy) else ""))
  cat(sprintf("  %-10s", "Model"))
  for (m in metrics_to_show) cat(sprintf(" %10s", m))
  cat("\n  ", strrep("-", 52), "\n", sep = "")
  for (i in seq_len(nrow(ranking))) {
    cat(sprintf("  %-10s", ranking$model[i]))
    for (m in metrics_to_show)
      cat(sprintf(" %10s", .fmt(ranking[[m]][i], 4)))
    if (i == 1L) cat("  ← best")
    cat("\n")
  }
  cat(strrep("═", 62), "\n\n")

  # ── Comparison plot ─────────────────────────────────────────────
  model_pal <- c(
    LSTM   = .PAL$recursive, GRU    = .PAL$direct,
    RNN    = "#984ea3",       ARIMA  = .PAL$arima,
    ARFIMA = .PAL$arfima
  )

  plot_long <- ranking %>%
    select(model, all_of(metrics_to_show)) %>%
    pivot_longer(-model, names_to = "metric", values_to = "value") %>%
    filter(!is.na(value)) %>%
    mutate(metric = factor(metric, levels = metrics_to_show),
           model  = factor(model,  levels = ranking$model))

  comp_plot <- ggplot(plot_long, aes(model, value, fill = model)) +
    geom_col(width = 0.6, show.legend = FALSE) +
    geom_text(aes(label = round(value, 3)),
              vjust = -0.4, size = 3.0) +
    facet_wrap(~metric, scales = "free_y", nrow = 1L) +
    scale_fill_manual(values = model_pal,
                      breaks = names(model_pal)) +
    labs(title    = "Unified Model Comparison — OOS CV Metrics",
         subtitle = sprintf("Ranked by %s (lower is better)", rank_by),
         x = NULL, y = "Value") +
    theme_minimal(base_size = 11) +
    theme(plot.title    = element_text(face = "bold", size = 13,
                                       hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, colour = "grey40"),
          axis.text.x   = element_text(angle = 30, hjust = 1))

  print(comp_plot)
  list(ranking = ranking, results = results, plot = comp_plot)
}


# ================================================================
# EXAMPLE USAGE
# ================================================================

# ── Uncomment to run ─────────────────────────────────────────────
#
# source("auto_rnn_framework.R")
# source("auto_rnn_s3_methods.R")
# source("auto_rnn_extended.R")   # this file
# library(zoo)
#
# df <- data.frame(
#   date  = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
#   value = as.numeric(AirPassengers)
# )
#
# # ── RNN model ──────────────────────────────────────────────────
# result <- auto_lstm(df, horizon = 12L, strategy = "both",
#                     run_bo = TRUE, run_cv = TRUE, cv_initial = 84L)
#
# # Out-of-sample fitted values (per fold, per horizon step)
# oos <- fitted_oos(result, strategy = "both")
# head(oos)
#
# # predict() for new/extended forecasts
# fc_new <- predict(result, horizon = 6L, strategy = "recursive", level = 90)
# print(fc_new)
#
# # predicted() — tidy OOS actual vs predicted
# pred_df <- predicted(result, strategy = "both")
# agg_df  <- predicted(result, aggregate = TRUE)
#
# # Full CV performance report
# perf <- cv_performance(result, strategy = "both", by_horizon = TRUE)
# perf$aggregate    # mean ± SD table
# perf$by_horizon   # metrics broken down by h step
# plot_cv_performance(perf)
#
# # ── Auto-ARIMA ─────────────────────────────────────────────────
# arima_result <- auto_arima_ts(df, horizon = 12L, run_cv = TRUE)
# print(arima_result)
# summary(arima_result)
# predict(arima_result, horizon = 6L)
# fitted(arima_result)
# predicted(arima_result, aggregate = TRUE)
# cv_performance(arima_result)
# plot(arima_result)
#
# # ── Auto-ARFIMA ────────────────────────────────────────────────
# arfima_result <- auto_arfima_ts(df, horizon = 12L, run_cv = TRUE)
# print(arfima_result)
# summary(arfima_result)
# predict(arfima_result)
# predicted(arfima_result, aggregate = TRUE)
# plot(arfima_result)
#
# # ── Unified comparison: all 5 models ───────────────────────────
# comp <- compare_all_models(
#   data         = df,
#   rnn_models   = c("lstm","gru","rnn"),
#   run_arima    = TRUE,
#   run_arfima   = TRUE,
#   horizon      = 12L,
#   rank_by      = "RMSE",
#   rnn_strategy = "recursive",
#   cv_initial   = 84L,
#   bo_init      = 3L,
#   bo_iter      = 8L,
#   verbose      = 0L
# )
#
# comp$ranking   # ranked comparison table
# comp$results$LSTM$forecast_recursive
# comp$results$ARIMA$forecast
# comp$results$ARFIMA$d_hat
