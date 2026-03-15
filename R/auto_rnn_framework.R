# ================================================================
# AutoRNN Framework (torch)
# Generalised Time Series Forecasting: LSTM · GRU · RNN
# ================================================================
#
# Single entry points:
#   auto_lstm()  →  auto_rnn_torch(model_type = "lstm", ...)
#   auto_gru()   →  auto_rnn_torch(model_type = "gru",  ...)
#   auto_rnn()   →  auto_rnn_torch(model_type = "rnn",  ...)
#
# All three share:
#   • Stacked recurrent architecture  (nn_lstm / nn_gru / nn_rnn)
#   • Bayesian hyperparameter optimisation (rBayesianOptimization)
#   • Recursive  multi-step forecasting (nextla-style)
#   • Direct     multi-step forecasting (one model per horizon)
#   • rolling_origin cross-validation  (rsample) – OOS evaluation
#   • Model comparison across all three architectures
#
# ================================================================
# Installation (run once):
#   install.packages(c("torch","tidyverse","rsample",
#                      "zoo","rBayesianOptimization"))
#   torch::install_torch()   # downloads libtorch (~500 MB)
# ================================================================

suppressPackageStartupMessages({
  library(torch)
  library(tidyverse)
  library(rsample)
  library(zoo)
  library(rBayesianOptimization)
})


# ================================================================
# SECTION 0 — CONSTANTS & VALID MODEL TYPES
# ================================================================

.VALID_MODEL_TYPES <- c("lstm", "gru", "rnn")

#' Validate model_type argument
.check_model_type <- function(model_type) {
  model_type <- tolower(trimws(model_type))
  if (!model_type %in% .VALID_MODEL_TYPES)
    stop(sprintf("model_type must be one of: %s",
                 paste(.VALID_MODEL_TYPES, collapse = ", ")))
  model_type
}


# ================================================================
# SECTION 1 — UTILITY HELPERS  (shared by all model types)
# ================================================================

#' Z-score scaler
make_scaler <- function(x) list(mean = mean(x, na.rm = TRUE),
                                 sd   = sd(x,   na.rm = TRUE))
scale_x     <- function(x, sc) (x - sc$mean) / sc$sd
unscale_x   <- function(x, sc)  x * sc$sd   + sc$mean

#' Infer dominant date frequency (returns string for seq(..., by =))
infer_frequency <- function(dates) {
  med <- median(as.numeric(diff(sort(dates))))
  dplyr::case_when(
    med <= 1  ~ "day",
    med <= 8  ~ "week",
    med <= 32 ~ "month",
    med <= 93 ~ "quarter",
    TRUE      ~ "year"
  )
}

#' Regression metrics
calc_metrics <- function(actual, predicted) {
  r <- actual - predicted
  list(RMSE = sqrt(mean(r^2)),
       MAE  = mean(abs(r)),
       MAPE = mean(abs(r / actual)) * 100)
}

#' Build supervised matrix for RECURSIVE strategy
#'   Returns list(X = matrix[n, lags], y = numeric[n])
make_supervised <- function(x, lags) {
  n <- length(x);  nr <- n - lags
  X <- matrix(NA_real_, nr, lags);  y <- numeric(nr)
  for (i in seq_len(nr)) { X[i,] <- x[i:(i+lags-1)];  y[i] <- x[i+lags] }
  list(X = X, y = y)
}

#' Build supervised matrix for DIRECT strategy (target h steps ahead)
make_supervised_direct <- function(x, lags, h) {
  n <- length(x);  nr <- n - lags - h + 1L
  if (nr <= 0L) stop("Not enough data for direct strategy with these settings.")
  X <- matrix(NA_real_, nr, lags);  y <- numeric(nr)
  for (i in seq_len(nr)) { X[i,] <- x[i:(i+lags-1)];  y[i] <- x[i+lags+h-1L] }
  list(X = X, y = y)
}

#' Auto-detect the best torch device
get_device <- function() {
  if (cuda_is_available())          "cuda"
  else if (backends_mps_is_available()) "mps"
  else                               "cpu"
}


# ================================================================
# SECTION 2 — TORCH DATASET
# ================================================================

TimeSeriesDataset <- dataset(
  name = "TimeSeriesDataset",

  initialize = function(X_mat, y_vec) {
    # X: [batch, seq_len, 1]   y: [batch, 1]
    self$X <- torch_tensor(X_mat,           dtype = torch_float())$unsqueeze(3L)
    self$y <- torch_tensor(as.numeric(y_vec), dtype = torch_float())$unsqueeze(2L)
  },

  .getitem = function(i) list(x = self$X[i,,], y = self$y[i,]),
  .length  = function()  dim(self$X)[1L]
)


# ================================================================
# SECTION 3 — GENERALISED STACKED RNN MODULE
# ================================================================

#' StackedRNN  — single nn_module supporting LSTM / GRU / RNN
#'
#' Architecture:
#'   Recurrent block (num_layers stacked, batch_first=TRUE)
#'     ↓  [last hidden state: batch × hidden_size]
#'   Dropout(dropout)
#'   Linear(hidden_size → dense_units) → ReLU
#'   Dropout(dropout/2)
#'   Linear(dense_units → 1)
#'
#' @param model_type  "lstm" | "gru" | "rnn"
#' @param input_size  Number of input features (1 for univariate)
#' @param hidden_size Units per recurrent layer
#' @param num_layers  Number of stacked recurrent layers
#' @param dropout     Dropout rate (between layers + after block)
#' @param dense_units Units in the FC head
#' @param rnn_nonlinearity  "tanh" or "relu" (only used when model_type="rnn")
StackedRNN <- nn_module(
  classname = "StackedRNN",

  initialize = function(model_type      = "lstm",
                        input_size      = 1L,
                        hidden_size     = 64L,
                        num_layers      = 2L,
                        dropout         = 0.1,
                        dense_units     = 32L,
                        rnn_nonlinearity = "tanh") {

    self$model_type  <- tolower(model_type)
    self$hidden_size <- as.integer(hidden_size)
    self$num_layers  <- as.integer(num_layers)

    inter_dropout <- if (num_layers > 1L) dropout else 0.0

    self$rnn <- switch(
      self$model_type,

      lstm = nn_lstm(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        batch_first = TRUE,
        dropout     = inter_dropout
      ),

      gru  = nn_gru(
        input_size  = input_size,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        batch_first = TRUE,
        dropout     = inter_dropout
      ),

      rnn  = nn_rnn(
        input_size   = input_size,
        hidden_size  = hidden_size,
        num_layers   = num_layers,
        nonlinearity = rnn_nonlinearity,
        batch_first  = TRUE,
        dropout      = inter_dropout
      ),

      stop(sprintf("Unsupported model_type: '%s'", model_type))
    )

    self$drop1  <- nn_dropout(p = dropout)
    self$fc1    <- nn_linear(hidden_size, dense_units)
    self$relu   <- nn_relu()
    self$drop2  <- nn_dropout(p = dropout / 2)
    self$fc_out <- nn_linear(dense_units, 1L)
  },

  forward = function(x) {
    # x : [batch, seq_len, input_size]
    rnn_out <- self$rnn(x)   # list: (output, hidden)  or  (output, (h,c))

    # Last time-step from output tensor: [batch, seq_len, hidden_size]
    out_tensor <- rnn_out[[1L]]
    last_step  <- out_tensor[, dim(out_tensor)[2L], ]  # [batch, hidden_size]

    last_step              %>%
      self$drop1()         %>%
      self$fc1()           %>%
      self$relu()          %>%
      self$drop2()         %>%
      self$fc_out()           # [batch, 1]
  }
)


# ================================================================
# SECTION 4 — GENERIC TRAINING LOOP
# ================================================================

#' Train a StackedRNN model  (LSTM / GRU / RNN)
#'
#' @param X_train,y_train   Training data (matrix, numeric).
#' @param X_val,y_val       Validation data (or NULL).
#' @param lags              Sequence length.
#' @param model_type        "lstm" | "gru" | "rnn".
#' @param hidden_size       Recurrent hidden units.
#' @param num_layers        Stacked recurrent layers.
#' @param dropout           Dropout rate.
#' @param dense_units       FC head units.
#' @param rnn_nonlinearity  "tanh" | "relu"  (RNN only).
#' @param lr                Adam learning rate.
#' @param epochs            Max epochs.
#' @param batch_size        Mini-batch size.
#' @param patience          Early-stopping patience.
#' @param lr_factor         LR reduction factor on plateau.
#' @param lr_patience       Patience for LR reduction.
#' @param grad_clip         Max gradient norm (NULL = no clipping).
#' @param verbose           0 = silent, 1 = periodic updates.
#' @param device            "cpu" | "cuda" | "mps".
#'
#' @return list(model = StackedRNN, history = data.frame)
train_rnn_model <- function(X_train,
                            y_train,
                            X_val            = NULL,
                            y_val            = NULL,
                            lags,
                            model_type       = "lstm",
                            hidden_size      = 64L,
                            num_layers       = 2L,
                            dropout          = 0.1,
                            dense_units      = 32L,
                            rnn_nonlinearity = "tanh",
                            lr               = 1e-3,
                            epochs           = 150L,
                            batch_size       = 32L,
                            patience         = 20L,
                            lr_factor        = 0.5,
                            lr_patience      = NULL,
                            grad_clip        = 1.0,
                            verbose          = 0L,
                            device           = get_device()) {

  model_type  <- .check_model_type(model_type)
  if (is.null(lr_patience)) lr_patience <- max(5L, patience %/% 3L)

  # ── Build model ───────────────────────────────────────────────
  model <- StackedRNN(
    model_type       = model_type,
    input_size       = 1L,
    hidden_size      = as.integer(hidden_size),
    num_layers       = as.integer(num_layers),
    dropout          = dropout,
    dense_units      = as.integer(dense_units),
    rnn_nonlinearity = rnn_nonlinearity
  )$to(device = device)

  optimizer <- optim_adam(model$parameters, lr = lr)
  scheduler <- lr_reduce_on_plateau(optimizer,
                                    mode     = "min",
                                    factor   = lr_factor,
                                    patience = lr_patience,
                                    min_lr   = 1e-7,
                                    verbose  = FALSE)
  loss_fn   <- nn_mse_loss()

  # ── DataLoaders ───────────────────────────────────────────────
  train_dl <- dataloader(TimeSeriesDataset(X_train, y_train),
                         batch_size = batch_size, shuffle = FALSE)
  use_val  <- !is.null(X_val) && length(y_val) > 0
  if (use_val)
    val_dl <- dataloader(TimeSeriesDataset(X_val, y_val),
                         batch_size = batch_size, shuffle = FALSE)

  # ── Training state ────────────────────────────────────────────
  history          <- data.frame(epoch      = integer(),
                                 train_loss = double(),
                                 val_loss   = double())
  best_val_loss    <- Inf
  best_state       <- NULL
  patience_counter <- 0L

  for (epoch in seq_len(epochs)) {

    # Train pass
    model$train()
    tr_loss_sum <- 0.0;  n_tr <- 0L
    coro::loop(for (b in train_dl) {
      optimizer$zero_grad()
      pred <- model(b$x$to(device = device))
      loss <- loss_fn(pred, b$y$to(device = device))
      loss$backward()
      if (!is.null(grad_clip))
        nn_utils_clip_grad_norm_(model$parameters, max_norm = grad_clip)
      optimizer$step()
      tr_loss_sum <- tr_loss_sum + loss$item();  n_tr <- n_tr + 1L
    })
    train_loss <- tr_loss_sum / n_tr

    # Validation pass
    val_loss <- NA_real_
    if (use_val) {
      model$eval()
      vl_sum <- 0.0;  n_vl <- 0L
      with_no_grad({
        coro::loop(for (b in val_dl) {
          p      <- model(b$x$to(device = device))
          vl_sum <- vl_sum + loss_fn(p, b$y$to(device = device))$item()
          n_vl   <- n_vl + 1L
        })
      })
      val_loss <- vl_sum / n_vl
      scheduler$step(val_loss)

      if (val_loss < best_val_loss - 1e-6) {
        best_val_loss    <- val_loss
        best_state       <- lapply(model$state_dict(), function(t) t$clone())
        patience_counter <- 0L
      } else {
        patience_counter <- patience_counter + 1L
      }
    }

    history <- rbind(history,
                     data.frame(epoch = epoch,
                                train_loss = train_loss,
                                val_loss   = val_loss))

    if (verbose > 0L && epoch %% max(1L, epochs %/% 10L) == 0L)
      cat(sprintf("  [%s] Epoch %4d/%d | train=%.5f | val=%.5f\n",
                  toupper(model_type), epoch, epochs, train_loss,
                  ifelse(is.na(val_loss), 0, val_loss)))

    if (use_val && patience_counter >= patience) {
      if (verbose > 0L)
        cat(sprintf("  Early stop at epoch %d  (best val=%.5f)\n",
                    epoch, best_val_loss))
      break
    }
  }

  if (!is.null(best_state)) model$load_state_dict(best_state)
  list(model = model, history = history)
}


# ================================================================
# SECTION 5 — GENERIC INFERENCE
# ================================================================

#' Run a trained StackedRNN on an R matrix, return numeric vector
predict_rnn <- function(model, X_mat, device = get_device()) {
  model$eval()
  X_t <- torch_tensor(X_mat, dtype = torch_float())$
    unsqueeze(3L)$to(device = device)
  with_no_grad({ out <- model(X_t) })
  as.numeric(out$squeeze(2L)$cpu())
}


# ================================================================
# SECTION 6 — BAYESIAN HYPERPARAMETER OPTIMISATION
# ================================================================

#' Bayesian optimisation of StackedRNN hyperparameters
#'
#' @param scaled_series   Normalised numeric vector.
#' @param lags            Fixed lag window.
#' @param model_type      "lstm" | "gru" | "rnn".
#' @param val_fraction    Validation fraction inside BO.
#' @param n_iter          BO optimisation iterations.
#' @param init_points     Random exploration rounds.
#' @param epochs          Max epochs per BO trial.
#' @param patience        EarlyStopping patience per trial.
#' @param device          Torch device.
#' @param seed            RNG seed.
#'
#' @return list(best_params, bo_result)
bayesian_optimize_rnn <- function(scaled_series,
                                  lags         = 12L,
                                  model_type   = "lstm",
                                  val_fraction = 0.20,
                                  n_iter       = 15L,
                                  init_points  = 5L,
                                  epochs       = 80L,
                                  patience     = 12L,
                                  device       = get_device(),
                                  seed         = 42L) {
  set.seed(seed)
  model_type <- .check_model_type(model_type)

  cat(sprintf(
    "\n── Bayesian HP Optimisation [%s] | device: %s | init: %d | iter: %d\n",
    toupper(model_type), device, init_points, n_iter
  ))

  sv <- make_supervised(scaled_series, lags)
  n  <- nrow(sv$X);  nv <- floor(n * val_fraction);  nt <- n - nv
  X_tr <- sv$X[1:nt,      , drop = FALSE];  y_tr <- sv$y[1:nt]
  X_vl <- sv$X[(nt+1):n,  , drop = FALSE];  y_vl <- sv$y[(nt+1):n]

  # BO objective: maximise Score = −best_val_loss
  objective_fn <- function(hidden_size, num_layers, dropout,
                           dense_units, lr_log10, batch_size_log2) {
    hp <- list(
      hidden_size  = as.integer(round(hidden_size)),
      num_layers   = as.integer(round(num_layers)),
      dropout      = dropout,
      dense_units  = as.integer(round(dense_units)),
      lr           = 10^lr_log10,
      batch_size   = as.integer(2^round(batch_size_log2))
    )
    tryCatch({
      res    <- train_rnn_model(
        X_train     = X_tr, y_train = y_tr,
        X_val       = X_vl, y_val   = y_vl,
        lags        = lags,
        model_type  = model_type,
        hidden_size = hp$hidden_size,
        num_layers  = hp$num_layers,
        dropout     = hp$dropout,
        dense_units = hp$dense_units,
        lr          = hp$lr,
        epochs      = epochs,
        batch_size  = hp$batch_size,
        patience    = patience,
        verbose     = 0L,
        device      = device
      )
      best_vl <- min(res$history$val_loss, na.rm = TRUE)
      list(Score = -best_vl, Pred = 0)
    }, error = function(e) list(Score = -Inf, Pred = 0))
  }

  bounds <- list(
    hidden_size     = c(16L,  128L),
    num_layers      = c(1L,   3L),
    dropout         = c(0.0,  0.4),
    dense_units     = c(16L,  128L),
    lr_log10        = c(-4.0, -2.0),   # 1e-4 … 1e-2
    batch_size_log2 = c(3.0,  6.0)     # 8 … 64
  )

  bo <- BayesianOptimization(
    FUN         = objective_fn,
    bounds      = bounds,
    init_points = init_points,
    n_iter      = n_iter,
    acq         = "ucb",
    kappa       = 2.576,
    verbose     = TRUE
  )

  bp <- bo$Best_Par
  best_params <- list(
    hidden_size  = as.integer(round(bp["hidden_size"])),
    num_layers   = as.integer(round(bp["num_layers"])),
    dropout      = bp["dropout"],
    dense_units  = as.integer(round(bp["dense_units"])),
    lr           = 10^bp["lr_log10"],
    batch_size   = as.integer(2^round(bp["batch_size_log2"]))
  )

  cat(sprintf("\n── [%s] Best hyperparameters ─────────────────────────────\n",
              toupper(model_type)))
  str(best_params)
  list(best_params = best_params, bo_result = bo)
}


# ================================================================
# SECTION 7 — RECURSIVE FORECASTING  (nextla-style)
# ================================================================

#' Multi-step recursive forecast
#'
#' @param model       Trained StackedRNN (on device).
#' @param last_window Numeric vector length `lags` (scaled).
#' @param horizon     Steps ahead.
#' @param scaler      list(mean, sd) for inverse transform.
#' @param device      Torch device.
#'
#' @return Numeric vector length `horizon` (original scale).
recursive_forecast <- function(model, last_window, horizon, scaler,
                               device = get_device()) {
  model$eval()
  window <- last_window
  preds  <- numeric(horizon)

  for (h in seq_len(horizon)) {
    X_t <- torch_tensor(matrix(window, nrow = 1L), dtype = torch_float())$
      unsqueeze(3L)$to(device = device)
    with_no_grad({ p <- model(X_t) })
    preds[h] <- as.numeric(p$item())
    window   <- c(window[-1L], preds[h])
  }
  unscale_x(preds, scaler)
}


# ================================================================
# SECTION 8 — DIRECT FORECASTING  (H separate models)
# ================================================================

#' Train H independent StackedRNN models (one per horizon step)
#'
#' @param scaled_series Full normalised training series.
#' @param lags          Lag window.
#' @param horizon       Total steps ahead.
#' @param hp            Named list of hyperparameters from BO.
#' @param model_type    "lstm" | "gru" | "rnn".
#' @param epochs, patience, val_fraction, device, verbose  training params.
#'
#' @return List of H trained StackedRNN modules.
train_direct_models <- function(scaled_series,
                                lags,
                                horizon,
                                hp,
                                model_type   = "lstm",
                                epochs       = 150L,
                                patience     = 20L,
                                val_fraction = 0.15,
                                device       = get_device(),
                                verbose      = 0L) {

  model_type <- .check_model_type(model_type)
  cat(sprintf(
    "\n── Direct models [%s] — training h = 1 … %d [device: %s]\n",
    toupper(model_type), horizon, device
  ))

  models <- vector("list", horizon)

  for (h in seq_len(horizon)) {
    cat(sprintf("  h = %2d / %d\r", h, horizon))
    sv <- make_supervised_direct(scaled_series, lags, h)
    n  <- nrow(sv$X);  nv <- floor(n * val_fraction);  nt <- n - nv

    res <- train_rnn_model(
      X_train     = sv$X[1:nt,     , drop = FALSE],
      y_train     = sv$y[1:nt],
      X_val       = if (nv > 0) sv$X[(nt+1):n,, drop = FALSE] else NULL,
      y_val       = if (nv > 0) sv$y[(nt+1):n] else NULL,
      lags        = lags,
      model_type  = model_type,
      hidden_size = hp$hidden_size,
      num_layers  = hp$num_layers,
      dropout     = hp$dropout,
      dense_units = hp$dense_units,
      lr          = hp$lr,
      epochs      = epochs,
      batch_size  = hp$batch_size,
      patience    = patience,
      verbose     = verbose,
      device      = device
    )
    models[[h]] <- res$model
  }
  cat("\n")
  models
}

#' Generate direct forecasts from H trained models
#'
#' @param direct_models List of H StackedRNN modules.
#' @param last_window   Numeric vector length `lags` (scaled).
#' @param scaler        list(mean, sd).
#' @param device        Torch device.
#'
#' @return Numeric vector length H (original scale).
direct_forecast <- function(direct_models, last_window, scaler,
                            device = get_device()) {
  X_mat <- matrix(last_window, nrow = 1L)
  preds <- sapply(direct_models, function(m)
    predict_rnn(m, X_mat, device)[1L])
  unscale_x(preds, scaler)
}


# ================================================================
# SECTION 9 — ROLLING ORIGIN CROSS-VALIDATION
# ================================================================

#' Out-of-sample evaluation with rolling_origin
#'
#' @param data         Data frame with columns `date` and `value`.
#' @param lags         Lag window.
#' @param horizon      Assessment window per fold.
#' @param hp           Named list of best hyperparameters.
#' @param model_type   "lstm" | "gru" | "rnn".
#' @param initial      Min training observations per fold.
#' @param skip         Observations skipped between origins.
#' @param cumulative   Expanding (TRUE) or sliding (FALSE) window.
#' @param strategy     "recursive" | "direct" | "both".
#' @param epochs, patience, val_fraction  training params.
#' @param scale        Z-score normalise.
#' @param device       Torch device.
#' @param verbose      0 = silent.
#'
#' @return list(cv_metrics, summary, fold_forecasts)
rolling_origin_eval <- function(data,
                                lags         = 12L,
                                horizon      = 12L,
                                hp,
                                model_type   = "lstm",
                                initial      = 60L,
                                skip         = 0L,
                                cumulative   = TRUE,
                                strategy     = c("both","recursive","direct"),
                                epochs       = 150L,
                                patience     = 20L,
                                val_fraction = 0.15,
                                scale        = TRUE,
                                device       = get_device(),
                                verbose      = 0L) {

  strategy   <- match.arg(strategy)
  model_type <- .check_model_type(model_type)

  splits <- rolling_origin(data,
                           initial    = initial,
                           assess     = horizon,
                           skip       = skip,
                           cumulative = cumulative)

  cat(sprintf(
    "\n── Rolling Origin CV [%s] | %d folds | horizon=%d | strategy=%s\n",
    toupper(model_type), nrow(splits), horizon, strategy
  ))

  fold_results <- vector("list", nrow(splits))

  for (i in seq_len(nrow(splits))) {
    cat(sprintf("  Fold %d / %d\r", i, nrow(splits)))

    tr_df  <- analysis(splits$splits[[i]])
    ts_df  <- assessment(splits$splits[[i]])
    tv     <- tr_df$value
    av     <- ts_df$value[seq_len(min(horizon, nrow(ts_df)))]

    sc         <- make_scaler(tv)
    tr_sc      <- if (scale) scale_x(tv, sc) else tv
    last_win   <- tail(tr_sc, lags)
    fold_row   <- list(fold = i, n_train = length(tv), actual = av)

    # ── Recursive ──────────────────────────────────────────────
    if (strategy %in% c("recursive","both")) {
      sv  <- make_supervised(tr_sc, lags)
      n   <- nrow(sv$X);  nv <- floor(n * val_fraction);  nt <- n - nv

      res <- train_rnn_model(
        X_train     = sv$X[1:nt,    , drop=FALSE], y_train = sv$y[1:nt],
        X_val       = if (nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
        y_val       = if (nv>0) sv$y[(nt+1):n] else NULL,
        lags        = lags, model_type = model_type,
        hidden_size = hp$hidden_size, num_layers  = hp$num_layers,
        dropout     = hp$dropout,    dense_units = hp$dense_units,
        lr          = hp$lr,         epochs      = epochs,
        batch_size  = hp$batch_size, patience    = patience,
        verbose     = verbose,       device      = device
      )
      fc <- recursive_forecast(res$model, last_win, horizon, sc, device)
      m  <- calc_metrics(av, fc[seq_along(av)])
      fold_row$recursive_forecast <- fc
      fold_row$RMSE_recursive     <- m$RMSE
      fold_row$MAE_recursive      <- m$MAE
      fold_row$MAPE_recursive     <- m$MAPE
    }

    # ── Direct ─────────────────────────────────────────────────
    if (strategy %in% c("direct","both")) {
      dm <- train_direct_models(tr_sc, lags, horizon, hp, model_type,
                                epochs, patience, val_fraction, device, verbose)
      fc <- direct_forecast(dm, last_win, sc, device)
      m  <- calc_metrics(av, fc[seq_along(av)])
      fold_row$direct_forecast <- fc
      fold_row$RMSE_direct     <- m$RMSE
      fold_row$MAE_direct      <- m$MAE
      fold_row$MAPE_direct     <- m$MAPE
    }

    fold_results[[i]] <- fold_row
  }
  cat("\n")

  # ── Aggregate metrics ─────────────────────────────────────────
  cv_metrics <- dplyr::bind_rows(lapply(fold_results, function(r) {
    row <- tibble(fold = r$fold, n_train = r$n_train)
    for (nm in c("RMSE_recursive","MAE_recursive","MAPE_recursive",
                 "RMSE_direct",   "MAE_direct",   "MAPE_direct")) {
      if (!is.null(r[[nm]])) row[[nm]] <- r[[nm]]
    }
    row
  }))

  num_cols <- setdiff(names(cv_metrics)[sapply(cv_metrics, is.numeric)], "fold")
  summary_stats <- cv_metrics                                        %>%
    summarise(across(all_of(num_cols),
                     list(mean = mean, sd = sd),
                     .names = "{.col}__{.fn}"))                      %>%
    pivot_longer(everything(),
                 names_to  = c("metric","stat"),
                 names_sep = "__")                                   %>%
    pivot_wider(names_from = stat, values_from = value)

  cat(sprintf("\n── [%s] Aggregated CV Metrics ──────────────────────────────\n",
              toupper(model_type)))
  print(summary_stats, n = Inf)

  list(cv_metrics     = cv_metrics,
       summary        = summary_stats,
       fold_forecasts = fold_results)
}


# ================================================================
# SECTION 10 — MASTER FUNCTION: auto_rnn_torch()
# ================================================================

#' AutoRNN: Generalised time series forecasting (LSTM / GRU / RNN)
#'
#' @param data            Data frame with date + value columns.
#' @param date_col        Date column name.
#' @param value_col       Value column name.
#' @param model_type      "lstm" | "gru" | "rnn"  (case-insensitive).
#' @param horizon         Forecast steps ahead.
#' @param lags            Lag window (NULL = auto: max(12, ceil(h*1.5))).
#' @param scale           Z-score normalise (strongly recommended).
#' @param strategy        "recursive" | "direct" | "both".
#' @param device          "cpu"|"cuda"|"mps"|NULL (auto-detect).
#' @param run_bo          Run Bayesian HP Optimisation.
#' @param bo_init         BO random exploration rounds.
#' @param bo_iter         BO optimisation iterations.
#' @param bo_epochs       Max epochs per BO trial.
#' @param bo_patience     EarlyStopping patience in BO trials.
#' @param final_epochs    Epochs for final model(s).
#' @param final_patience  EarlyStopping patience for final model(s).
#' @param run_cv          Run rolling_origin CV evaluation.
#' @param cv_initial      Min training obs per CV fold (NULL = auto).
#' @param cv_skip         Obs skipped between fold origins.
#' @param cv_cumulative   Expanding (TRUE) or sliding (FALSE) window.
#' @param val_fraction    Internal val split (BO, final, CV).
#' @param seed            RNG seed (R + torch).
#' @param verbose         0 = silent, 1 = progress.
#'
#' @return Named list:
#'   model_type, forecast_recursive, forecast_direct,
#'   best_params, bo_result, cv_results,
#'   model_recursive, direct_models,
#'   history_recursive, scaler, data
auto_rnn_torch <- function(data,
                           date_col        = "date",
                           value_col       = "value",
                           model_type      = "lstm",
                           horizon         = 12L,
                           lags            = NULL,
                           scale           = TRUE,
                           strategy        = c("both","recursive","direct"),
                           device          = NULL,
                           run_bo          = TRUE,
                           bo_init         = 5L,
                           bo_iter         = 15L,
                           bo_epochs       = 80L,
                           bo_patience     = 12L,
                           final_epochs    = 200L,
                           final_patience  = 25L,
                           run_cv          = TRUE,
                           cv_initial      = NULL,
                           cv_skip         = 0L,
                           cv_cumulative   = TRUE,
                           val_fraction    = 0.15,
                           seed            = 42L,
                           verbose         = 1L) {

  set.seed(seed)
  torch_manual_seed(seed)
  model_type <- .check_model_type(model_type)
  strategy   <- match.arg(strategy)
  if (is.null(device)) device <- get_device()

  # ── 1. Validate & clean ───────────────────────────────────────
  stopifnot(is.data.frame(data),
            date_col  %in% names(data),
            value_col %in% names(data))

  df <- data                                       %>%
    rename(date  = !!sym(date_col),
           value = !!sym(value_col))               %>%
    arrange(date)                                  %>%
    select(date, value)

  if (anyNA(df$value)) {
    warning("Missing values detected — interpolating with zoo::na.approx.")
    df$value <- zoo::na.approx(df$value, na.rm = FALSE)
  }

  if (is.null(lags)) lags <- max(12L, as.integer(ceiling(horizon * 1.5)))

  cat(sprintf(
    "\nAuto%s (torch) | n=%d | horizon=%d | lags=%d | strategy=%s | device=%s\n",
    toupper(model_type), nrow(df), horizon, lags, strategy, device
  ))

  # ── 2. Scale ──────────────────────────────────────────────────
  sc     <- make_scaler(df$value)
  scaled <- if (scale) scale_x(df$value, sc) else df$value

  # ── 3. Default HPs ────────────────────────────────────────────
  default_hp <- list(hidden_size = 64L, num_layers  = 2L, dropout     = 0.1,
                     dense_units = 32L, lr          = 1e-3, batch_size  = 32L)

  # ── 4. Bayesian Optimisation ──────────────────────────────────
  bo_result <- NULL

  if (run_bo) {
    bo_out    <- bayesian_optimize_rnn(
      scaled_series = scaled,  lags       = lags,
      model_type    = model_type, val_fraction = val_fraction,
      n_iter        = bo_iter, init_points  = bo_init,
      epochs        = bo_epochs, patience   = bo_patience,
      device        = device,  seed         = seed
    )
    best_hp   <- bo_out$best_params
    bo_result <- bo_out$bo_result
  } else {
    cat("\nUsing default hyperparameters (BO skipped).\n")
    best_hp <- default_hp
  }

  # ── 5. Rolling Origin CV ──────────────────────────────────────
  cv_results <- NULL

  if (run_cv) {
    if (is.null(cv_initial))
      cv_initial <- max(lags + horizon + 10L,
                        as.integer(nrow(df) * 0.5))
    cv_results <- rolling_origin_eval(
      data         = df, lags = lags, horizon = horizon, hp = best_hp,
      model_type   = model_type, initial = cv_initial, skip = cv_skip,
      cumulative   = cv_cumulative, strategy = strategy,
      epochs       = final_epochs, patience = final_patience,
      val_fraction = val_fraction, scale = scale,
      device       = device, verbose = 0L
    )
  }

  # ── 6. Final models on full series ───────────────────────────
  cat(sprintf(
    "\n── Final [%s] training on full series ──────────────────────\n",
    toupper(model_type)
  ))

  sv_full  <- make_supervised(scaled, lags)
  n        <- nrow(sv_full$X);  nv <- floor(n * val_fraction);  nt <- n - nv
  X_tr     <- sv_full$X[1:nt,    , drop=FALSE];  y_tr <- sv_full$y[1:nt]
  X_vl     <- if (nv>0) sv_full$X[(nt+1):n,, drop=FALSE] else NULL
  y_vl     <- if (nv>0) sv_full$y[(nt+1):n] else NULL
  last_win <- tail(scaled, lags)
  freq     <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by = freq, length.out = horizon + 1L)[-1L]

  # Shared CI builder
  make_ci <- function(mdl, Xm, yv, fc) {
    pred   <- predict_rnn(mdl, Xm, device)
    res_sd <- sd(unscale_x(yv, sc) - unscale_x(pred, sc))
    ci_w   <- qnorm(0.975) * res_sd * sqrt(seq_len(horizon))
    list(lower_95 = fc - ci_w, upper_95 = fc + ci_w)
  }

  # ── 6a. Recursive final model ─────────────────────────────────
  model_recursive   <- NULL
  history_recursive <- NULL
  forecast_rec_df   <- NULL

  if (strategy %in% c("recursive","both")) {
    cat("  [Recursive] Training…\n")
    res <- train_rnn_model(
      X_train     = X_tr, y_train = y_tr, X_val = X_vl, y_val = y_vl,
      lags        = lags, model_type  = model_type,
      hidden_size = best_hp$hidden_size, num_layers  = best_hp$num_layers,
      dropout     = best_hp$dropout,    dense_units = best_hp$dense_units,
      lr          = best_hp$lr,         epochs      = final_epochs,
      batch_size  = best_hp$batch_size, patience    = final_patience,
      verbose     = verbose,            device      = device
    )
    model_recursive   <- res$model
    history_recursive <- res$history

    fc  <- recursive_forecast(model_recursive, last_win, horizon, sc, device)
    ci  <- make_ci(model_recursive, X_tr, y_tr, fc)
    forecast_rec_df <- tibble(date     = fut_dates, forecast = fc,
                              lower_95 = ci$lower_95, upper_95 = ci$upper_95,
                              strategy = "recursive")
    cat("  [Recursive] Done.\n")
  }

  # ── 6b. Direct final models ───────────────────────────────────
  direct_models   <- NULL
  forecast_dir_df <- NULL

  if (strategy %in% c("direct","both")) {
    cat("  [Direct] Training H models…\n")
    direct_models <- train_direct_models(
      scaled_series = scaled, lags = lags, horizon = horizon, hp = best_hp,
      model_type    = model_type, epochs = final_epochs, patience = final_patience,
      val_fraction  = val_fraction, device = device, verbose = 0L
    )
    fc    <- direct_forecast(direct_models, last_win, sc, device)
    sv1   <- make_supervised_direct(scaled, lags, 1L)
    nt1   <- nrow(sv1$X) - floor(nrow(sv1$X) * val_fraction)
    p1    <- predict_rnn(direct_models[[1]], sv1$X[1:nt1,,drop=FALSE], device)
    ci_w1 <- qnorm(0.975) * sd(unscale_x(sv1$y[1:nt1],sc) - unscale_x(p1,sc)) *
             sqrt(seq_len(horizon))
    forecast_dir_df <- tibble(date     = fut_dates, forecast = fc,
                              lower_95 = fc - ci_w1, upper_95 = fc + ci_w1,
                              strategy = "direct")
    cat("  [Direct] Done.\n")
  }

  cat(sprintf("\n✓ Auto%s (torch) complete.\n", toupper(model_type)))

  structure(
    list(model_type         = model_type,
         forecast_recursive = forecast_rec_df,
         forecast_direct    = forecast_dir_df,
         best_params        = best_hp,
         bo_result          = bo_result,
         cv_results         = cv_results,
         model_recursive    = model_recursive,
         direct_models      = direct_models,
         history_recursive  = history_recursive,
         scaler             = sc,
         data               = df),
    class = c(paste0("auto_", model_type), "auto_rnn")
  )
}


# ================================================================
# SECTION 11 — CONVENIENCE WRAPPERS
# ================================================================

#' AutoLSTM: LSTM-based time series forecasting
#' Thin wrapper around auto_rnn_torch(model_type = "lstm")
#' @inheritParams auto_rnn_torch
auto_lstm <- function(data, ...) auto_rnn_torch(data, model_type = "lstm", ...)

#' AutoGRU: GRU-based time series forecasting
#' Thin wrapper around auto_rnn_torch(model_type = "gru")
#' @inheritParams auto_rnn_torch
auto_gru  <- function(data, ...) auto_rnn_torch(data, model_type = "gru",  ...)

#' AutoRNN: Vanilla RNN-based time series forecasting
#' Thin wrapper around auto_rnn_torch(model_type = "rnn")
#' @inheritParams auto_rnn_torch
auto_rnn  <- function(data, ...) auto_rnn_torch(data, model_type = "rnn",  ...)


# ================================================================
# SECTION 12 — MODEL COMPARISON UTILITY
# ================================================================

#' Compare LSTM, GRU, and RNN on the same dataset
#'
#' Trains all three model types with identical settings and returns
#' a ranked comparison table and per-model result objects.
#'
#' @param data       Data frame with date + value columns.
#' @param models     Character vector of model types to compare.
#' @param rank_by    Metric to rank by: "RMSE" | "MAE" | "MAPE".
#' @param ...        Additional arguments forwarded to auto_rnn_torch().
#'
#' @return list(comparison = tibble, results = named list)
compare_models <- function(data,
                           models   = c("lstm","gru","rnn"),
                           rank_by  = "RMSE",
                           ...) {
  models  <- sapply(models, .check_model_type)
  results <- setNames(vector("list", length(models)), models)

  for (mt in models) {
    cat(sprintf("\n════════════════════════════════════════\n"))
    cat(sprintf("  Training Auto%s\n", toupper(mt)))
    cat(sprintf("════════════════════════════════════════\n"))
    results[[mt]] <- auto_rnn_torch(data, model_type = mt, ...)
  }

  # Extract CV summary per model
  rows <- lapply(models, function(mt) {
    r   <- results[[mt]]
    smry <- r$cv_results$summary
    if (is.null(smry)) return(tibble(model = toupper(mt)))

    # Pull mean of RMSE_recursive and RMSE_direct if available
    get_mean <- function(metric_name) {
      row <- smry[smry$metric == metric_name, ]
      if (nrow(row) == 0) return(NA_real_)
      row$mean
    }

    tibble(
      model          = toupper(mt),
      RMSE_recursive = get_mean("RMSE_recursive"),
      MAE_recursive  = get_mean("MAE_recursive"),
      MAPE_recursive = get_mean("MAPE_recursive"),
      RMSE_direct    = get_mean("RMSE_direct"),
      MAE_direct     = get_mean("MAE_direct"),
      MAPE_direct    = get_mean("MAPE_direct")
    )
  })

  comparison <- dplyr::bind_rows(rows)

  # Rank by chosen metric (recursive first, then direct)
  rank_col <- paste0(rank_by, "_recursive")
  if (rank_col %in% names(comparison) && !all(is.na(comparison[[rank_col]])))
    comparison <- comparison %>% arrange(.data[[rank_col]])

  cat("\n════════════ Model Comparison ════════════\n")
  print(comparison)
  cat(sprintf("  Ranked by %s (recursive strategy)\n", rank_by))

  list(comparison = comparison, results = results)
}


# ================================================================
# SECTION 13 — PLOTTING HELPERS
# ================================================================

#' Plot forecast(s) against historical series
#'
#' @param result  Output of auto_rnn_torch() / auto_lstm() / etc.
#' @param last_n  Plot only the last N historical observations (NULL = all).
#' @param title   Plot title (NULL = auto-generated).
plot_forecasts <- function(result, last_n = NULL, title = NULL) {
  if (is.null(title))
    title <- sprintf("Auto%s (torch) Forecast", toupper(result$model_type))

  hist_df <- result$data
  if (!is.null(last_n)) hist_df <- tail(hist_df, last_n)

  fc_all <- dplyr::bind_rows(
    Filter(Negate(is.null),
           list(result$forecast_recursive, result$forecast_direct))
  )
  pal <- c(recursive = "#2166ac", direct = "#d6604d")

  p <- ggplot() +
    geom_line(data = hist_df, aes(date, value),
              colour = "grey30", linewidth = 0.85) +
    geom_vline(xintercept = as.numeric(max(hist_df$date)),
               linetype = "dashed", colour = "grey60")

  if (nrow(fc_all) > 0) {
    p <- p +
      geom_ribbon(data = fc_all,
                  aes(date, ymin = lower_95, ymax = upper_95, fill = strategy),
                  alpha = 0.18) +
      geom_line(data  = fc_all,
                aes(date, forecast, colour = strategy),
                linewidth = 1.0, linetype = "dashed") +
      geom_point(data = fc_all,
                 aes(date, forecast, colour = strategy), size = 2) +
      scale_colour_manual(values = pal) +
      scale_fill_manual(values = pal)
  }

  p + labs(title    = title,
           subtitle = "Shaded band = 95% CI  |  Dashed = forecast",
           x = NULL, y = "Value", colour = "Strategy", fill = "Strategy") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "top")
}

#' Plot rolling-origin CV metric per fold
#'
#' @param result  Output of auto_rnn_torch().
#' @param metric  "RMSE" | "MAE" | "MAPE".
plot_cv_metrics <- function(result, metric = "RMSE") {
  cv   <- result$cv_results$cv_metrics
  cols <- grep(metric, names(cv), value = TRUE)
  if (length(cols) == 0L) stop("Metric not found in CV results.")

  cv                                                              %>%
    select(fold, all_of(cols))                                   %>%
    pivot_longer(-fold, names_to = "strategy", values_to = "value") %>%
    mutate(strategy = gsub(paste0(metric, "_"), "", strategy))   %>%
    ggplot(aes(fold, value, colour = strategy)) +
    geom_line(linewidth = 0.9) + geom_point(size = 2.5) +
    scale_colour_manual(values = c(recursive = "#2166ac", direct = "#d6604d")) +
    labs(title = sprintf("[%s] Rolling Origin CV — %s per Fold",
                         toupper(result$model_type), metric),
         x = "Fold", y = metric, colour = "Strategy") +
    theme_minimal(base_size = 13) +
    theme(legend.position = "top")
}

#' Plot training loss curves for the recursive final model
#'
#' @param result  Output of auto_rnn_torch().
plot_training_history <- function(result) {
  h <- result$history_recursive
  if (is.null(h)) {
    message("No training history available.")
    return(invisible(NULL))
  }
  h                                                               %>%
    select(epoch, Training = train_loss, Validation = val_loss)  %>%
    pivot_longer(-epoch, names_to = "split", values_to = "MSE")  %>%
    filter(!is.na(MSE))                                          %>%
    ggplot(aes(epoch, MSE, colour = split)) +
    geom_line(linewidth = 0.9) +
    scale_colour_manual(
      values = c(Training = "#2166ac", Validation = "#d6604d")) +
    labs(title    = sprintf("[%s] Training History — Recursive Model",
                            toupper(result$model_type)),
         subtitle = "EarlyStopping active on validation loss",
         x = "Epoch", y = "MSE Loss", colour = NULL) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "top")
}

#' Bar chart comparing model types across metrics
#'
#' @param comparison_result  Output of compare_models().
#' @param metric             "RMSE" | "MAE" | "MAPE".
#' @param strategy           "recursive" | "direct".
plot_model_comparison <- function(comparison_result,
                                  metric   = "RMSE",
                                  strategy = "recursive") {
  col <- paste0(metric, "_", strategy)
  df  <- comparison_result$comparison

  if (!col %in% names(df)) stop(sprintf("Column '%s' not found.", col))

  df                                                     %>%
    select(model, value = all_of(col))                   %>%
    filter(!is.na(value))                                %>%
    mutate(model = factor(model, levels = model[order(value)])) %>%
    ggplot(aes(model, value, fill = model)) +
    geom_col(width = 0.55, show.legend = FALSE) +
    geom_text(aes(label = round(value, 3)),
              vjust = -0.4, size = 3.5) +
    scale_fill_manual(values = c(
      LSTM = "#2166ac", GRU = "#4dac26", RNN = "#d6604d")) +
    labs(title    = sprintf("Model Comparison — %s (%s strategy)", metric, strategy),
         subtitle = "Lower is better",
         x = NULL, y = metric) +
    theme_minimal(base_size = 13)
}


# ================================================================
# SECTION 14 — EXAMPLE USAGE
# ================================================================

# ── Uncomment blocks to run ───────────────────────────────────────
#
# library(zoo)
#
# df <- data.frame(
#   date  = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
#   value = as.numeric(AirPassengers)
# )
#
# # ── A) Run a single model type ──────────────────────────────────
# result_lstm <- auto_lstm(
#   data = df, horizon = 12L, strategy = "both",
#   run_bo = TRUE, bo_init = 5L, bo_iter = 10L,
#   run_cv = TRUE, cv_initial = 84L, verbose = 1L
# )
#
# result_gru <- auto_gru(
#   data = df, horizon = 12L, strategy = "both",
#   run_bo = TRUE, bo_init = 5L, bo_iter = 10L,
#   run_cv = TRUE, cv_initial = 84L, verbose = 1L
# )
#
# result_rnn <- auto_rnn(
#   data = df, horizon = 12L, strategy = "both",
#   run_bo = TRUE, bo_init = 5L, bo_iter = 10L,
#   run_cv = TRUE, cv_initial = 84L, verbose = 1L
# )
#
# # ── B) Inspect results ─────────────────────────────────────────
# print(result_lstm$forecast_recursive)
# print(result_lstm$best_params)
# print(result_lstm$cv_results$summary)
#
# plot_forecasts(result_lstm, last_n = 60)
# plot_cv_metrics(result_lstm, metric = "RMSE")
# plot_training_history(result_lstm)
#
# # ── C) Compare all three architectures ─────────────────────────
# comp <- compare_models(
#   data     = df,
#   models   = c("lstm","gru","rnn"),
#   rank_by  = "RMSE",
#   horizon  = 12L,
#   strategy = "recursive",
#   run_bo   = TRUE, bo_init = 3L, bo_iter = 8L,
#   run_cv   = TRUE, cv_initial = 84L,
#   verbose  = 0L
# )
#
# print(comp$comparison)
# plot_model_comparison(comp, metric = "RMSE", strategy = "recursive")
#
# # Access individual model results from comparison
# print(comp$results$lstm$forecast_recursive)
# print(comp$results$gru$cv_results$summary)
