# ================================================================
# torchforecast.R  —  Unified Time Series Forecasting Framework
# Version 6.0  (single file)
# ================================================================
#
# KEY CHANGES v6.0  (relative to v5.0)
# ────────────────────────────────────────────────────────────────
#   ① CV engine  →  modeltime.resample
#       time_series_cv()            replaces rolling_origin()
#       modeltime_fit_resamples()   replaces manual tsCV loops
#       resample_accuracy()         unified accuracy extraction
#       Works for ALL model types: deep learning (LSTM/GRU/RNN/
#       N-BEATS) and statistical (ARIMA, ETS, ARFIMA, HAR, GARCH)
#       through a unified modeltime bridge layer.
#
#   ② Bayesian HP optimisation  →  ParBayesianOptimization
#       bayesOpt()      replaces rBayesianOptimization::BayesianOptimization()
#       getBestPars()   extracts optimal hyperparameters
#       Supports parallel = TRUE natively via doParallel backend
#       Also handles integer bounds and acq = "ucb"/"ei"/"poi"
#
#   ③ Fast numerics  →  matrixStats + RcppRoll
#       matrixStats::colMeans2() / colSds()   fast column stats
#       RcppRoll::roll_mean()                 fast rolling windows
#       matrixStats::rowMins() / rowMaxs()    used in N-BEATS
#       These replace all base R loops for matrix scaling and
#       HAR-RV lag construction
#
# MODELS (unchanged from v5)
# ────────────────────────────────────────────────────────────────
#   Deep Learning:  LSTM · GRU · RNN · N-BEATS
#   Classical:      Auto-ARIMA · Auto-ETS · Auto-ARFIMA
#   Econometric:    HAR-RV · GARCH (rugarch)
#
# INSTALLATION (run once)
# ────────────────────────────────────────────────────────────────
#   install.packages(c(
#     "torch","coro","tidyverse","timetk","zoo",
#     "ParBayesianOptimization","forecast","modeltime",
#     "modeltime.resample","matrixStats","RcppRoll",
#     "fracdiff","tseries","rugarch","patchwork","scales",
#     "sandwich","lmtest","future","furrr","parallelly",
#     "doFuture","doParallel"
#   ))
#   torch::install_torch()
#
# USAGE
# ────────────────────────────────────────────────────────────────
#   source("torchforecast.R")
#   tf_setup_parallel()
#   lstm <- auto_lstm(train_df, horizon = 12L)
#   predict(lstm, horizon = 12L)            # future
#   predict(lstm, new_data = test_df)       # OOS fitted
#   mt  <- tf_modeltime_table(lstm, arima_res)
#   cal <- tf_calibrate(list(LSTM=lstm, ARIMA=arima_res), test_df)
#   tf_accuracy(cal)
# ================================================================

suppressPackageStartupMessages({
  library(torch);                library(coro)
  library(tidyverse);            library(timetk)
  library(zoo);                  library(forecast)
  library(modeltime);            library(modeltime.resample)
  library(ParBayesianOptimization)
  library(matrixStats);          library(RcppRoll)
  library(fracdiff);             library(tseries)
  library(rugarch);              library(patchwork)
  library(scales);               library(sandwich)
  library(lmtest);               library(future)
  library(furrr);                library(parallelly)
  library(doFuture);             library(doParallel)
})


# ================================================================
# ██  PART 0 — GLOBAL CONSTANTS & HELPERS
#              (matrixStats + RcppRoll replace base R loops)
# ================================================================

.VALID_RNN <- c("lstm","gru","rnn")
.PAL <- list(
  recursive="#2166ac", direct="#d6604d",    nbeats="#1d91c0",
  arima="#1b7837",     arfima="#8c510a",     har="#e08214",
  garch="#762a83",     ets="#4dac26",
  actual="grey20",     neutral="grey60",     train="#35978f",
  val="#762a83",
  LSTM="#2166ac",  GRU="#d6604d",    RNN="#984ea3",   NBEATS="#1d91c0",
  ARIMA="#1b7837", ETS="#4dac26",    ARFIMA="#8c510a",
  HAR="#e08214",   GARCH="#762a83"
)

`%||%`        <- function(a,b) if(!is.null(a)) a else b
.fmt          <- function(x,d=4) formatC(x,digits=d,format="f",flag=" ")
.box          <- function(ch="═",w=66) strrep(ch,w)
.hdr          <- function(txt,ch="─",w=66){
  pad <- max(0L,w-nchar(txt)-4L)
  sprintf("%s  %s  %s",strrep(ch,2),txt,strrep(ch,pad))
}
.tic          <- function() proc.time()["elapsed"]
.toc          <- function(t0) as.numeric(proc.time()["elapsed"]-t0)
.fmt_elapsed  <- function(s){
  if(is.na(s)||is.null(s)) return("NA")
  if(s<60)     sprintf("%.1fs",s)
  else if(s<3600) sprintf("%.1fmin",s/60)
  else sprintf("%.2fh",s/3600)
}

# ── matrixStats-based fast scaler ──────────────────────────────
# For a single column vector: keep list(mean, sd) for unscaling.
# For a matrix X: col-wise centering/scaling via matrixStats.

make_scaler   <- function(x) list(mean=mean(x,na.rm=TRUE),sd=sd(x,na.rm=TRUE))
scale_x       <- function(x,sc) (x-sc$mean)/sc$sd
unscale_x     <- function(x,sc)  x*sc$sd+sc$mean

#' Scale each column of a matrix using matrixStats (fast, no loops)
#' Returns list(X_scaled, col_means, col_sds) for later unscaling
scale_matrix_cols <- function(X){
  cm  <- matrixStats::colMeans2(X, na.rm=TRUE)
  csd <- matrixStats::colSds(X,   na.rm=TRUE)
  csd[csd < .Machine$double.eps] <- 1.0   # guard against zero-sd columns
  X_scaled <- sweep(sweep(X, 2L, cm, "-"), 2L, csd, "/")
  list(X_scaled=X_scaled, col_means=cm, col_sds=csd)
}

#' Reverse column-wise scaling
unscale_matrix_cols <- function(X_scaled, col_means, col_sds){
  sweep(sweep(X_scaled, 2L, col_sds, "*"), 2L, col_means, "+")
}

# ── RcppRoll-based fast rolling mean ──────────────────────────
# Replaces zoo::rollmean and manual loops in HAR feature building.
# roll_mean returns a vector of same length as x with NA fill.

.roll_mean_right <- function(x, n){
  if(length(x) < n) return(rep(NA_real_, length(x)))
  RcppRoll::roll_mean(x, n=n, align="right", fill=NA_real_)
}

# ── Supervised matrix builders ─────────────────────────────────
# Use matrixStats for the column-stats path; loop construction
# is kept (unavoidable for row-offset indexing) but matrix ops
# are vectorised.

make_supervised <- function(x, lags){
  n  <- length(x); nr <- n - lags
  if(nr <= 0L) stop("Series too short for given lags.")
  # Build lag matrix: col j = x shifted by (lags - j + 1)
  X <- matrix(NA_real_, nr, lags)
  for(j in seq_len(lags)) X[, j] <- x[j:(j + nr - 1L)]
  y <- x[(lags + 1L):n]
  list(X=X, y=y)
}

make_supervised_direct <- function(x, lags, h){
  n  <- length(x); nr <- n - lags - h + 1L
  if(nr <= 0L) stop("Not enough data for direct strategy.")
  X <- matrix(NA_real_, nr, lags)
  for(j in seq_len(lags)) X[, j] <- x[j:(j + nr - 1L)]
  y <- x[(lags + h):(n)]
  list(X=X, y=y)
}

.safe_metrics <- function(actual, predicted){
  r <- actual - predicted
  list(
    RMSE  = sqrt(mean(r^2,   na.rm=TRUE)),
    MAE   = mean(abs(r),     na.rm=TRUE),
    MAPE  = if(any(actual==0, na.rm=TRUE)) NA_real_
            else mean(abs(r/actual)*100, na.rm=TRUE),
    SMAPE = mean(200*abs(r)/(abs(actual)+abs(predicted)), na.rm=TRUE),
    MBE   = mean(r, na.rm=TRUE)
  )
}
.check_arch <- function(arch){
  al <- tolower(trimws(arch)); valid <- c(.VALID_RNN,"nbeats")
  if(!al %in% valid) stop(sprintf("arch must be one of: %s",paste(valid,collapse=", ")))
  al
}
.base_theme <- function()
  theme_minimal(base_size=11)+
  theme(plot.title=element_text(face="bold",size=11),
        plot.subtitle=element_text(size=9,colour="grey45"),
        legend.position="top",legend.key.size=unit(0.4,"cm"))
.bt <- .base_theme

get_device <- function(){
  if(cuda_is_available())              "cuda"
  else if(backends_mps_is_available()) "mps"
  else                                 "cpu"
}
infer_frequency <- function(dates){
  med <- median(as.numeric(diff(sort(dates))))
  dplyr::case_when(med<=1~"day",med<=8~"week",med<=32~"month",
                   med<=93~"quarter",TRUE~"year")
}
.ts_freq <- function(dates){
  med <- median(as.numeric(diff(sort(dates))))
  dplyr::case_when(med<=1~365L,med<=8~52L,med<=32~12L,med<=93~4L,TRUE~1L)
}
.df_to_ts <- function(df){
  f <- .ts_freq(df$date)
  s <- c(as.integer(format(min(df$date),"%Y")),
         if(f==12L) as.integer(format(min(df$date),"%m"))
         else if(f==4L) as.integer(ceiling(as.integer(format(min(df$date),"%m"))/3))
         else 1L)
  ts(df$value, start=s, frequency=f)
}

.make_pred_tbl <- function(date, value, lo, hi, model_desc, key="prediction"){
  tibble(date=date, .value=value, .conf_lo=lo, .conf_hi=hi,
         .model_desc=model_desc, .key=key)
}


# ================================================================
# ██  PART 1 — PARALLEL BACKEND
# ================================================================

#' @export
tf_setup_parallel <- function(workers=NULL,
                               backend=c("multisession","multicore","cluster","sequential"),
                               gc=TRUE, verbose=TRUE){
  backend <- match.arg(backend)
  workers <- if(is.null(workers)||identical(workers,"auto"))
    max(1L, parallelly::availableCores(which="system")-1L)
  else if(identical(workers,"max")) parallelly::availableCores()
  else {
    w <- as.integer(workers); a <- parallelly::availableCores()
    if(w>a){warning(sprintf("Only %d cores.",a)); a} else w
  }
  if(backend=="multicore"){
    if(.Platform$OS.type=="windows"){warning("→ multisession");backend<-"multisession"}
    else if(isNamespaceLoaded("torch")){message("torch → multisession");backend<-"multisession"}
  }
  if(backend=="sequential"||workers<=1L){
    future::plan(future::sequential); workers <- 1L
  } else {
    future::plan(switch(backend,
      multisession=future::multisession,
      multicore=future::multicore,
      cluster=future::cluster),
      workers=workers, gc=gc)
  }
  # Register doFuture for ParBayesianOptimization's parallel back-end
  doFuture::registerDoFuture()
  options(tf.workers=workers, tf.backend=backend)
  if(verbose){
    cat(sprintf("\n── torchforecast parallel ──────────────────────────────────\n"))
    cat(sprintf("   Backend: %-16s  Workers: %d\n", backend, workers))
    cat(sprintf("   BO backend: doFuture (ParBayesianOptimization)\n"))
    cat(sprintf("   CV backend: modeltime.resample\n"))
    cat(strrep("─",60),"\n\n")
  }
  invisible(workers)
}

#' @export
tf_reset_parallel <- function(){
  future::plan(future::sequential)
  doParallel::stopImplicitCluster()
  options(tf.workers=1L, tf.backend="sequential")
  message("torchforecast: sequential."); invisible(NULL)
}
.n_workers   <- function() getOption("tf.workers", default=1L)
.is_parallel <- function() .n_workers() > 1L


# ================================================================
# ██  PART 2 — rnn_config()
# ================================================================

#' @export
rnn_config <- function(
    hidden_size=64L, num_layers=2L, dropout=0.1, dense_units=32L,
    rnn_nonlinearity="tanh",
    lr=1e-3, batch_size=32L, final_epochs=200L, final_patience=25L,
    lr_factor=0.5, lr_patience=NULL, grad_clip=1.0, val_fraction=0.15,
    run_bo=TRUE, bo_init=5L, bo_iter=15L, bo_epochs=80L, bo_patience=12L,
    bo_acq="ucb", bo_kappa=2.576, bo_bounds=NULL,
    run_cv=TRUE, cv_initial=NULL, cv_window=NULL, cv_skip=1L,
    cv_holdout_frac=0.20,
    lags=NULL, scale=TRUE, seed=42L, verbose=1L){
  .vbn <- c("hidden_size","num_layers","dropout","dense_units","lr_log10","batch_size_log2")
  if(!is.null(bo_bounds)){
    bad <- setdiff(names(bo_bounds), .vbn)
    if(length(bad)) stop(sprintf("Unknown bo_bounds: %s", paste(bad,collapse=",")))
    for(nm in names(bo_bounds)){b<-bo_bounds[[nm]]
      if(!is.numeric(b)||length(b)!=2||b[1]>=b[2])
        stop(sprintf("bo_bounds$%s: need c(lo,hi) with lo<hi",nm))}
  }
  structure(list(
    hidden_size=as.integer(hidden_size), num_layers=as.integer(num_layers),
    dropout=dropout, dense_units=as.integer(dense_units),
    rnn_nonlinearity=rnn_nonlinearity,
    lr=lr, batch_size=as.integer(batch_size),
    final_epochs=as.integer(final_epochs), final_patience=as.integer(final_patience),
    lr_factor=lr_factor, lr_patience=lr_patience, grad_clip=grad_clip,
    val_fraction=val_fraction,
    run_bo=run_bo, bo_init=as.integer(bo_init), bo_iter=as.integer(bo_iter),
    bo_epochs=as.integer(bo_epochs), bo_patience=as.integer(bo_patience),
    bo_acq=bo_acq, bo_kappa=bo_kappa, bo_bounds=bo_bounds,
    run_cv=run_cv, cv_initial=cv_initial, cv_window=cv_window, cv_skip=as.integer(cv_skip),
    cv_holdout_frac=cv_holdout_frac,
    lags=lags, scale=scale, seed=as.integer(seed), verbose=as.integer(verbose)
  ), class="rnn_config")
}

#' @export
print.rnn_config <- function(x,...){
  cat("\n",.box(),"\n  rnn_config  (v6)\n",.box(),"\n",sep="")
  cat(sprintf("  Arch  : hidden=%d | layers=%d | dropout=%.2f | dense=%d\n",
              x$hidden_size,x$num_layers,x$dropout,x$dense_units))
  cat(sprintf("  Train : lr=%.2e | batch=%d | epochs=%d | patience=%d\n",
              x$lr,x$batch_size,x$final_epochs,x$final_patience))
  cat(sprintf("  BO    : %s [ParBayesianOptimization]",if(x$run_bo)"ON" else "OFF"))
  if(x$run_bo) cat(sprintf(" init=%d | iter=%d | acq=%s",x$bo_init,x$bo_iter,x$bo_acq))
  cat("\n")
  cat(sprintf("  CV    : %s [modeltime.resample::time_series_cv] skip=%d | window=%s\n",
    if(x$run_cv)"ON" else "OFF", x$cv_skip,
    ifelse(is.null(x$cv_window),"expanding",x$cv_window)))
  invisible(x)
}

.resolve_cfg <- function(config, overrides){
  base <- rnn_config()
  if(!is.null(config)){
    if(!inherits(config,"rnn_config")) stop("config must be from rnn_config().")
    for(nm in names(config)) base[[nm]] <- config[[nm]]
  }
  for(nm in names(overrides)) if(!is.null(overrides[[nm]])) base[[nm]] <- overrides[[nm]]
  base
}
.default_bo_bounds <- list(
  hidden_size=c(16L,128L), num_layers=c(1L,3L), dropout=c(0.0,0.4),
  dense_units=c(16L,128L), lr_log10=c(-4.0,-2.0), batch_size_log2=c(3.0,6.0))
.resolve_bounds <- function(user){
  b <- .default_bo_bounds
  if(!is.null(user)) for(nm in names(user)) b[[nm]] <- user[[nm]]
  # ParBayesianOptimization uses numeric bounds; integers are rounded inside scorer
  lapply(b, function(v) as.numeric(v))
}


# ================================================================
# ██  PART 3 — TORCH DATASET
# ================================================================

TimeSeriesDataset <- dataset(
  name="TimeSeriesDataset",
  initialize=function(X_mat, y_vec){
    self$X <- torch_tensor(X_mat, dtype=torch_float())$unsqueeze(3L)
    self$y <- torch_tensor(as.numeric(y_vec), dtype=torch_float())$unsqueeze(2L)
  },
  .getitem=function(i) list(x=self$X[i,,], y=self$y[i,]),
  .length=function() dim(self$X)[1L]
)


# ================================================================
# ██  PART 4 — StackedRNN (LSTM / GRU / vanilla RNN)
# ================================================================

StackedRNN <- nn_module(
  classname="StackedRNN",
  initialize=function(model_type="lstm", input_size=1L, hidden_size=64L,
                      num_layers=2L, dropout=0.1, dense_units=32L,
                      rnn_nonlinearity="tanh"){
    self$model_type <- tolower(model_type)
    id <- if(num_layers>1L) dropout else 0.0
    self$rnn <- switch(self$model_type,
      lstm=nn_lstm(input_size,hidden_size,num_layers,batch_first=TRUE,dropout=id),
      gru =nn_gru (input_size,hidden_size,num_layers,batch_first=TRUE,dropout=id),
      rnn =nn_rnn (input_size,hidden_size,num_layers,nonlinearity=rnn_nonlinearity,
                   batch_first=TRUE,dropout=id),
      stop(sprintf("Unknown model_type: '%s'",model_type)))
    self$drop1 <- nn_dropout(p=dropout)
    self$fc1   <- nn_linear(hidden_size, dense_units)
    self$relu  <- nn_relu()
    self$drop2 <- nn_dropout(p=dropout/2)
    self$out   <- nn_linear(dense_units, 1L)
  },
  forward=function(x){
    o    <- self$rnn(x)[[1L]]
    last <- o[,dim(o)[2L],]
    last |> self$drop1() |> self$fc1() |> self$relu() |> self$drop2() |> self$out()
  }
)


# ================================================================
# ██  PART 5 — N-BEATS
# ================================================================

NBEATSBlock <- nn_module(
  classname="NBEATSBlock",
  initialize=function(input_size, theta_size, fc_width=512L, n_layers=4L,
                      basis_type="generic", horizon=12L){
    self$basis_type <- basis_type; self$theta_size <- theta_size
    self$input_size <- input_size; self$horizon    <- horizon
    layers <- list(); in_sz <- input_size
    for(i in seq_len(n_layers)){
      layers[[2L*i-1L]] <- nn_linear(in_sz, fc_width)
      layers[[2L*i]]    <- nn_relu(); in_sz <- fc_width
    }
    self$fc_stack  <- nn_sequential(!!!layers)
    self$theta_b   <- nn_linear(fc_width, theta_size, bias=FALSE)
    self$theta_f   <- nn_linear(fc_width, theta_size, bias=FALSE)
    if(basis_type=="generic"){
      self$backcast_basis <- nn_linear(theta_size, input_size, bias=FALSE)
      self$forecast_basis <- nn_linear(theta_size, horizon,    bias=FALSE)
    }
  },
  .trend_basis=function(T, degree, dev="cpu"){
    t    <- torch_arange(0,T-1L,dtype=torch_float(),device=dev)$unsqueeze(2L)/(T-1L)
    pows <- torch_arange(0,degree-1L,dtype=torch_float(),device=dev)$unsqueeze(1L)
    t$pow(pows)
  },
  .seasonality_basis=function(T, n_harmonics, dev="cpu"){
    t    <- torch_arange(0,T-1L,dtype=torch_float(),device=dev); cols <- list()
    for(i in seq_len(n_harmonics)){
      arg <- 2*pi*i*t/T
      cols[[2L*i-1L]] <- arg$cos(); cols[[2L*i]] <- arg$sin()
    }
    torch_stack(cols, dim=2L)
  },
  forward=function(x){
    dev <- x$device; h <- self$fc_stack(x)
    tb  <- self$theta_b(h); tf_ <- self$theta_f(h)
    if(self$basis_type=="generic"){
      backcast <- self$backcast_basis(tb)
      forecast <- self$forecast_basis(tf_)
    } else {
      deg <- self$theta_size%/%2L; n_h <- self$theta_size-deg
      Vb  <- self$.trend_basis(self$input_size,deg,dev$type)
      Vf  <- self$.trend_basis(self$horizon,   deg,dev$type)
      backcast <- torch_matmul(tb[,1:deg],Vb$t())
      forecast <- torch_matmul(tf_[,1:deg],Vf$t())
      if(n_h>0L){
        n_harm <- n_h%/%2L
        Sb <- self$.seasonality_basis(self$input_size,n_harm,dev$type)
        Sf <- self$.seasonality_basis(self$horizon,   n_harm,dev$type)
        backcast <- backcast + torch_matmul(tb[,(deg+1):self$theta_size], Sb$t())
        forecast <- forecast + torch_matmul(tf_[,(deg+1):self$theta_size],Sf$t())
      }
    }
    list(backcast=backcast, forecast=forecast)
  }
)

NBEATSModel <- nn_module(
  classname="NBEATSModel",
  initialize=function(input_size, horizon, n_stacks=2L, n_blocks=3L,
                      fc_width=512L, n_layers=4L, theta_size=NULL,
                      basis_type="generic"){
    self$input_size <- input_size; self$horizon <- horizon; self$basis_type <- basis_type
    if(is.null(theta_size))
      theta_size <- if(basis_type=="generic") 2L*horizon else max(3L,as.integer(log2(horizon))+1L)
    blocks <- list(); k <- 1L
    for(s in seq_len(n_stacks)) for(b in seq_len(n_blocks)){
      blocks[[k]] <- NBEATSBlock(input_size,theta_size,fc_width,n_layers,basis_type,horizon)
      k <- k+1L
    }
    self$blocks <- nn_module_list(blocks)
  },
  forward=function(x){
    residual <- x
    forecast <- torch_zeros(x$shape[1],self$horizon,dtype=x$dtype,device=x$device)
    for(i in seq_along(self$blocks)){
      out      <- self$blocks[[i]](residual)
      residual <- residual - out$backcast
      forecast <- forecast + out$forecast
    }
    forecast$unsqueeze(2L)
  }
)


# ================================================================
# ██  PART 6 — TRAINING LOOP (shared RNN + N-BEATS)
# ================================================================

train_deep_model <- function(X_train, y_train, X_val=NULL, y_val=NULL,
                              lags, arch="lstm",
                              hidden_size=64L, num_layers=2L, dropout=0.1,
                              dense_units=32L, rnn_nonlinearity="tanh",
                              nbeats_stacks=2L, nbeats_blocks=3L,
                              nbeats_width=512L, nbeats_layers=4L,
                              nbeats_theta=NULL, nbeats_basis="generic",
                              lr=1e-3, epochs=150L, batch_size=32L,
                              patience=20L, lr_factor=0.5, lr_patience=NULL,
                              grad_clip=1.0, horizon=1L, verbose=0L,
                              device=get_device()){
  t0 <- .tic(); if(is.null(lr_patience)) lr_patience <- max(5L,patience%/%3L)
  al <- tolower(arch); is_nb <- al=="nbeats"
  model <- if(!is_nb)
    StackedRNN(al,1L,as.integer(hidden_size),as.integer(num_layers),dropout,
               as.integer(dense_units),rnn_nonlinearity)$to(device=device)
  else
    NBEATSModel(lags,horizon,nbeats_stacks,nbeats_blocks,nbeats_width,
                nbeats_layers,nbeats_theta,nbeats_basis)$to(device=device)

  optimizer <- optim_adam(model$parameters, lr=lr)
  scheduler <- lr_reduce_on_plateau(optimizer, mode="min", factor=lr_factor,
                                     patience=lr_patience, min_lr=1e-7, verbose=FALSE)
  loss_fn   <- nn_mse_loss()

  make_dl <- function(Xm, yv){
    ds <- if(is_nb){
      dataset(
        initialize=function(X,y){
          self$X <- torch_tensor(X, dtype=torch_float())
          self$y <- torch_tensor(as.numeric(y), dtype=torch_float())$unsqueeze(2L)
        },
        .getitem=function(i) list(x=self$X[i,], y=self$y[i,]),
        .length=function() dim(self$X)[1L]
      )(Xm, yv)
    } else TimeSeriesDataset(Xm, yv)
    dataloader(ds, batch_size=batch_size, shuffle=FALSE)
  }

  tr_dl   <- make_dl(X_train, y_train)
  use_val <- !is.null(X_val)&&length(y_val)>0
  if(use_val) vl_dl <- make_dl(X_val, y_val)

  history <- data.frame(epoch=integer(), train_loss=double(), val_loss=double())
  best_vl <- Inf; best_st <- NULL; pat_ctr <- 0L

  for(ep in seq_len(epochs)){
    model$train(); tr_sum <- 0.0; n_tr <- 0L
    coro::loop(for(b in tr_dl){
      optimizer$zero_grad()
      xd   <- b$x$to(device=device)
      pred <- if(is_nb) model(xd)[,1L,] else model(xd)
      loss <- loss_fn(pred, b$y$to(device=device))
      loss$backward()
      if(!is.null(grad_clip)) nn_utils_clip_grad_norm_(model$parameters, max_norm=grad_clip)
      optimizer$step(); tr_sum <- tr_sum+loss$item(); n_tr <- n_tr+1L
    })
    tr_loss <- tr_sum/n_tr; vl_loss <- NA_real_
    if(use_val){
      model$eval(); vl_sum <- 0.0; n_vl <- 0L
      with_no_grad({
        coro::loop(for(b in vl_dl){
          xd   <- b$x$to(device=device)
          pred <- if(is_nb) model(xd)[,1L,] else model(xd)
          vl_sum <- vl_sum+loss_fn(pred, b$y$to(device=device))$item(); n_vl <- n_vl+1L
        })
      })
      vl_loss <- vl_sum/n_vl; scheduler$step(vl_loss)
      if(vl_loss < best_vl-1e-6){
        best_vl <- vl_loss
        best_st <- lapply(model$state_dict(), function(t) t$clone()); pat_ctr <- 0L
      } else pat_ctr <- pat_ctr+1L
    }
    history <- rbind(history, data.frame(epoch=ep, train_loss=tr_loss, val_loss=vl_loss))
    if(verbose>0L && ep%%max(1L,epochs%/%10L)==0L)
      cat(sprintf("  [%s] ep%4d | tr=%.5f | vl=%.5f\n", toupper(al), ep, tr_loss,
                  ifelse(is.na(vl_loss),0,vl_loss)))
    if(use_val && pat_ctr>=patience){if(verbose>0L) cat(sprintf("  Early stop ep%d\n",ep));break}
  }
  if(!is.null(best_st)) model$load_state_dict(best_st)
  list(model=model, history=history, elapsed_sec=.toc(t0))
}


# ================================================================
# ██  PART 7 — INFERENCE HELPERS
# ================================================================

predict_deep <- function(model, X_mat, device=get_device(), is_nbeats=FALSE){
  model$eval()
  X_t <- if(is_nbeats) torch_tensor(X_mat, dtype=torch_float())$to(device=device)
         else torch_tensor(X_mat, dtype=torch_float())$unsqueeze(3L)$to(device=device)
  with_no_grad({out <- model(X_t)})
  if(is_nbeats) as.numeric(out[,1L,]$squeeze()$cpu())
  else          as.numeric(out$squeeze(2L)$cpu())
}

recursive_forecast <- function(model, last_window, horizon, scaler, device=get_device()){
  model$eval(); window <- last_window; preds <- numeric(horizon)
  for(h in seq_len(horizon)){
    X_t <- torch_tensor(matrix(window,nrow=1L), dtype=torch_float())$
      unsqueeze(3L)$to(device=device)
    with_no_grad({p <- model(X_t)})
    preds[h] <- as.numeric(p$item()); window <- c(window[-1L], preds[h])
  }
  unscale_x(preds, scaler)
}

nbeats_forecast <- function(model, last_window, horizon, scaler, device=get_device()){
  model$eval()
  X_t <- torch_tensor(matrix(last_window,nrow=1L), dtype=torch_float())$to(device=device)
  with_no_grad({out <- model(X_t)})
  unscale_x(as.numeric(out$squeeze()$cpu())[seq_len(horizon)], scaler)
}

.rolling_oos <- function(model, train_sc, test_sc, lags, scaler, device, is_nb=FALSE){
  all_sc <- c(train_sc, test_sc); n_train <- length(train_sc); n_test <- length(test_sc)
  preds  <- numeric(n_test)
  for(i in seq_len(n_test)){
    ws  <- n_train+i-1L; wst <- ws-lags+1L
    if(wst<1L){preds[i]<-NA_real_;next}
    win <- all_sc[wst:ws]
    X_t <- torch_tensor(matrix(win,nrow=1L), dtype=torch_float())$
      unsqueeze(3L)$to(device=device)
    model$eval(); with_no_grad({p <- model(X_t)}); preds[i] <- as.numeric(p$item())
  }
  unscale_x(preds, scaler)
}

train_direct_models <- function(scaled_series, lags, horizon, hp, arch="lstm",
                                 epochs=150L, patience=20L, val_fraction=0.15,
                                 lr_factor=0.5, lr_patience=NULL, grad_clip=1.0,
                                 device=get_device(), verbose=0L, use_parallel=FALSE){
  cat(sprintf("\n── Direct [%s] h=1…%d %s\n", toupper(arch), horizon,
              if(use_parallel&&.is_parallel())"[parallel]" else ""))
  one_h <- function(h_step){
    sv <- make_supervised_direct(scaled_series, lags, h_step)
    n  <- nrow(sv$X); nv <- floor(n*val_fraction); nt <- n-nv
    r  <- train_deep_model(sv$X[1:nt,,drop=FALSE], sv$y[1:nt],
             if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
             if(nv>0) sv$y[(nt+1):n] else NULL,
             lags=lags, arch=arch, hidden_size=hp$hidden_size, num_layers=hp$num_layers,
             dropout=hp$dropout, dense_units=hp$dense_units, lr=hp$lr, epochs=epochs,
             batch_size=hp$batch_size, patience=patience, lr_factor=lr_factor,
             lr_patience=lr_patience, grad_clip=grad_clip, horizon=1L, verbose=0L, device=device)
    r$model
  }
  if(use_parallel&&.is_parallel())
    furrr::future_map(seq_len(horizon), function(h){
      suppressPackageStartupMessages(library(torch)); one_h(h)},
      .options=furrr::furrr_options(seed=TRUE, globals=FALSE), .progress=TRUE)
  else {
    models <- lapply(seq_len(horizon), function(h){
      cat(sprintf("  h=%2d/%d\r",h,horizon)); one_h(h)})
    cat("\n"); models
  }
}

direct_forecast <- function(direct_models, last_window, scaler, device=get_device()){
  X  <- matrix(last_window, nrow=1L)
  ps <- sapply(direct_models, function(m) predict_deep(m,X,device)[1L])
  unscale_x(ps, scaler)
}


# ================================================================
# ██  PART 8 — BAYESIAN HP OPTIMISATION  (ParBayesianOptimization)
# ================================================================
#
# Replaces rBayesianOptimization::BayesianOptimization() with
# ParBayesianOptimization::bayesOpt().
#
# Key differences from v5:
#   • bayesOpt() expects a scoring function that returns a list
#     with element "Score" (numeric, higher = better).
#   • Bounds are passed as a named list of c(lo,hi) numeric pairs.
#   • parallel = TRUE uses the currently-registered foreach
#     backend (doFuture, registered in tf_setup_parallel()).
#   • getBestPars() extracts the best hyperparameter set.
#   • initPoints sets the number of random initial evaluations.
#   • iters.n sets the number of GP-guided iterations.
#   • acqThresh / acq / kappa match the previous API semantics.
# ================================================================

.bo_optimize <- function(scaled_series, lags, arch, val_fraction,
                          n_iter, init_points, epochs, patience,
                          acq, kappa, bounds, device, seed,
                          horizon=1L, use_parallel=FALSE){
  t0 <- .tic(); set.seed(seed); arch_l <- tolower(arch)
  use_par <- use_parallel && .is_parallel()
  cat(sprintf("\n── ParBayesOpt [%s] | acq=%s | init=%d | iter=%d%s\n",
    toupper(arch), acq, init_points, n_iter,
    if(use_par) sprintf(" | workers=%d [doFuture]", .n_workers()) else " [sequential]"))

  # ── Train / validation split ──────────────────────────────────
  sv <- make_supervised(scaled_series, lags)
  n  <- nrow(sv$X); nv <- floor(n*val_fraction); nt <- n-nv
  # Use matrixStats for fast column mean/sd normalisation of the
  # supervised matrix before passing to the torch model
  X_all <- sv$X
  sc_m  <- scale_matrix_cols(X_all)
  X_all_sc <- sc_m$X_scaled
  Xt <- X_all_sc[1:nt,, drop=FALSE]; yt <- sv$y[1:nt]
  Xv <- X_all_sc[(nt+1):n,, drop=FALSE]; yv <- sv$y[(nt+1):n]

  # ── Scoring function for ParBayesianOptimization ──────────────
  # Must return a named list with element Score (numeric).
  # Higher Score = better. Here Score = negative val MSE.
  scoring_fn <- function(hidden_size, num_layers, dropout, dense_units,
                          lr_log10, batch_size_log2){
    hp_list <- list(
      hidden_size  = as.integer(round(hidden_size)),
      num_layers   = as.integer(round(num_layers)),
      dropout      = dropout,
      dense_units  = as.integer(round(dense_units)),
      lr           = 10^lr_log10,
      batch_size   = as.integer(2^round(batch_size_log2))
    )
    score <- tryCatch({
      suppressPackageStartupMessages(library(torch))
      r <- train_deep_model(Xt, yt, Xv, yv,
             lags=lags, arch=arch_l,
             hidden_size=hp_list$hidden_size, num_layers=hp_list$num_layers,
             dropout=hp_list$dropout, dense_units=hp_list$dense_units,
             lr=hp_list$lr, epochs=epochs, batch_size=hp_list$batch_size,
             patience=patience, horizon=horizon, verbose=0L, device=device)
      -min(r$history$val_loss, na.rm=TRUE)   # negative MSE → maximise
    }, error=function(e) -Inf)
    list(Score=score)
  }

  # ── Run ParBayesianOptimization::bayesOpt() ───────────────────
  # parallel=TRUE uses the registered doFuture backend automatically
  opt_obj <- tryCatch(
    ParBayesianOptimization::bayesOpt(
      FUN        = scoring_fn,
      bounds     = bounds,            # named list of c(lo,hi) numeric
      initPoints = init_points,
      iters.n    = n_iter,
      acq        = acq,               # "ucb", "ei", or "poi"
      kappa      = kappa,             # UCB exploration parameter
      parallel   = use_par,
      verbose    = 1L,
      gsPoints   = max(10L, n_iter*2L)
    ),
    error=function(e){
      warning(sprintf("bayesOpt failed (%s). Using random init.",e$message))
      NULL
    }
  )

  # ── Extract best hyperparameters ──────────────────────────────
  if(!is.null(opt_obj)){
    # getBestPars() returns a data.frame with one row
    bp   <- ParBayesianOptimization::getBestPars(opt_obj)
    best <- list(
      hidden_size  = as.integer(round(bp$hidden_size)),
      num_layers   = as.integer(round(bp$num_layers)),
      dropout      = bp$dropout,
      dense_units  = as.integer(round(bp$dense_units)),
      lr           = 10^bp$lr_log10,
      batch_size   = as.integer(2^round(bp$batch_size_log2))
    )
  } else {
    # Fallback: pick best from random init_points
    set.seed(seed)
    rnd_configs <- lapply(seq_len(init_points), function(i) list(
      hidden_size  = as.integer(round(runif(1,bounds$hidden_size[1],bounds$hidden_size[2]))),
      num_layers   = as.integer(round(runif(1,bounds$num_layers[1],bounds$num_layers[2]))),
      dropout      = runif(1,bounds$dropout[1],bounds$dropout[2]),
      dense_units  = as.integer(round(runif(1,bounds$dense_units[1],bounds$dense_units[2]))),
      lr           = 10^runif(1,bounds$lr_log10[1],bounds$lr_log10[2]),
      batch_size   = as.integer(2^round(runif(1,bounds$batch_size_log2[1],bounds$batch_size_log2[2])))))
    rnd_scores <- vapply(rnd_configs, function(hp){
      scoring_fn(hp$hidden_size,hp$num_layers,hp$dropout,hp$dense_units,
                 log10(hp$lr),log2(hp$batch_size))$Score
    }, numeric(1L))
    best <- rnd_configs[[which.max(rnd_scores)]]
  }

  cat(sprintf("\n── [%s] Best HP (ParBayesOpt) ──\n",toupper(arch))); str(best)
  list(best_params=best, bo_result=opt_obj, elapsed_sec=.toc(t0))
}


# ================================================================
# ██  PART 9 — CV ENGINE  (modeltime.resample)
# ================================================================
#
# Replaces forecast::tsCV() and the manual fold loop with:
#
#   timetk::time_series_cv()
#     → creates rsample-compatible resampling splits
#       cumulative = TRUE  → expanding window
#       cumulative = FALSE → sliding (fixed) window of size cv_window
#
#   modeltime.resample::modeltime_fit_resamples()
#     → fits a modeltime_table across all splits in parallel
#       (uses the doFuture backend registered in tf_setup_parallel)
#
#   modeltime.resample::resample_accuracy()
#     → extracts per-fold and average metrics from the resampled fits
#
# Deep learning models are wrapped in a minimal modeltime bridge
# so they participate in the standard resample workflow.
# Statistical models (ARIMA, ETS, ARFIMA, HAR, GARCH) are wrapped
# the same way — each split trains a fresh model on the analysis
# set and forecasts the assessment set.
# ================================================================

# ── Build a timetk time_series_cv resampling plan ──────────────
.make_resamples <- function(data, cv_initial, horizon, cv_skip, cv_window){
  # cumulative = TRUE  → expanding window (cv_window ignored)
  # cumulative = FALSE → sliding window of size cv_window
  cumulative <- is.null(cv_window)
  timetk::time_series_cv(
    data       = data,
    date_var   = date,
    initial    = cv_initial,
    assess     = horizon,
    skip       = cv_skip,
    cumulative = cumulative,
    slice_limit= if(!is.null(cv_window)) as.integer(ceiling((nrow(data)-cv_window)/cv_skip)) else NULL
  )
}

# ── OOS tibble builder from modeltime resample results ─────────
.resample_to_oos_tbl <- function(resample_results, strategy_name){
  # resample_results is the output of modeltime_fit_resamples()
  # .resample_results column contains per-fold modeltime_calibrated
  tryCatch({
    resample_results %>%
      modeltime.resample::modeltime_resample_accuracy(
        metric_set = yardstick::metric_set(
          yardstick::rmse, yardstick::mae, yardstick::mape, yardstick::smape)
      ) %>%
      mutate(strategy = strategy_name)
  }, error=function(e) {
    warning(sprintf(".resample_to_oos_tbl: %s", e$message))
    NULL
  })
}

# ── Metrics summary from resample accuracy tibble ──────────────
.resample_accuracy_summary <- function(acc_tbl){
  if(is.null(acc_tbl)||nrow(acc_tbl)==0L) return(NULL)
  # acc_tbl columns: .model_id, .model_desc, .metric, .estimate, [.resample_id]
  # We pivot to wide (RMSE, MAE, MAPE, SMAPE) then average across resamples
  acc_wide <- tryCatch(
    acc_tbl %>%
      select(any_of(c(".metric",".estimate","strategy",".resample_id"))) %>%
      group_by(.metric, strategy) %>%
      summarise(mean=mean(.estimate,na.rm=TRUE), sd=sd(.estimate,na.rm=TRUE), .groups="drop") %>%
      pivot_wider(names_from=.metric, values_from=mean, names_glue="{.metric}"),
    error=function(e) NULL)
  acc_wide
}

# ── Fit deep learning model inside a forecastfunction closure ──
# Used by the bridge that wraps DL results for modeltime.resample.
# This is the same closure pattern as v5's tsCV forecastfunction,
# but now the splits come from time_series_cv() instead of tsCV().

.deep_forecastfn <- function(hp, lags, arch_l, epochs, patience,
                              val_fraction, lr_factor, lr_patience,
                              grad_clip, scale, device, strategy){
  # Return a function(splits_analysis_df, splits_assessment_df) → predictions tibble
  function(analysis_df, assessment_df){
    suppressPackageStartupMessages(library(torch))
    y_tr   <- analysis_df$value
    n_tst  <- nrow(assessment_df)
    sc_par <- list(mean=mean(y_tr,na.rm=TRUE), sd=sd(y_tr,na.rm=TRUE))
    y_sc   <- if(scale)(y_tr-sc_par$mean)/sc_par$sd else y_tr
    last_w <- tail(y_sc, lags)

    if(strategy=="recursive"){
      sv <- make_supervised(y_sc, lags); n <- nrow(sv$X)
      nv <- floor(n*val_fraction); nt <- n-nv
      if(nt<5L) return(NULL)
      res <- train_deep_model(
        sv$X[1:nt,,drop=FALSE], sv$y[1:nt],
        if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
        if(nv>0) sv$y[(nt+1):n] else NULL,
        lags=lags, arch=arch_l, hidden_size=hp$hidden_size,
        num_layers=hp$num_layers, dropout=hp$dropout, dense_units=hp$dense_units,
        lr=hp$lr, epochs=epochs, batch_size=hp$batch_size, patience=patience,
        lr_factor=lr_factor, lr_patience=lr_patience, grad_clip=grad_clip,
        horizon=1L, verbose=0L, device=device)
      fc_sc <- numeric(n_tst)
      win   <- last_w
      for(i in seq_len(n_tst)){
        X_t <- torch_tensor(matrix(win,nrow=1L),dtype=torch_float())$
          unsqueeze(3L)$to(device=device)
        res$model$eval(); with_no_grad({p<-res$model(X_t)}); fc_sc[i]<-as.numeric(p$item())
        win <- c(win[-1L], fc_sc[i])
      }
      fc_vals <- fc_sc*sc_par$sd + sc_par$mean
    } else if(strategy=="nbeats"){
      sv <- make_supervised(y_sc, lags); n <- nrow(sv$X)
      nv <- floor(n*val_fraction); nt <- n-nv
      if(nt<5L) return(NULL)
      res <- train_deep_model(
        sv$X[1:nt,,drop=FALSE], sv$y[1:nt],
        if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
        if(nv>0) sv$y[(nt+1):n] else NULL,
        lags=lags, arch="nbeats", hidden_size=hp$hidden_size,
        num_layers=hp$num_layers, dropout=hp$dropout, dense_units=hp$dense_units,
        lr=hp$lr, epochs=epochs, batch_size=hp$batch_size, patience=patience,
        lr_factor=lr_factor, lr_patience=lr_patience, grad_clip=grad_clip,
        horizon=n_tst, verbose=0L, device=device)
      fc_vals <- nbeats_forecast(res$model, last_w, n_tst, sc_par, device)
    } else {
      return(NULL)  # direct not supported per-fold (too slow)
    }
    tibble(
      date      = assessment_df$date,
      actual    = assessment_df$value,
      predicted = fc_vals,
      residual  = assessment_df$value - fc_vals,
      strategy  = strategy
    )
  }
}

# ── Master CV function (modeltime.resample) ────────────────────
#' Run rolling-origin CV via modeltime.resample for deep learning models
#'
#' @param data         data frame with date + value columns
#' @param lags         input window (lags)
#' @param horizon      forecast horizon h
#' @param hp           best hyperparameter list
#' @param arch         architecture string
#' @param cv_initial   min training observations (timetk initial=)
#' @param cv_window    NULL = expanding; integer = sliding window size
#' @param cv_skip      skip between folds (timetk skip=)
#' @param strategy     "recursive" | "direct" | "both"
#' @param epochs,...   training settings
.resample_cv_deep <- function(data, lags, horizon, hp, arch,
                               cv_initial, cv_window, cv_skip, strategy,
                               epochs, patience, val_fraction, lr_factor,
                               lr_patience, grad_clip, scale, device, verbose,
                               use_parallel=FALSE){
  t0 <- .tic(); arch_l <- tolower(arch)
  cat(sprintf("\n── modeltime.resample [%s] | h=%d | initial=%s | window=%s | skip=%d\n",
    toupper(arch_l), horizon,
    cv_initial %||% "auto",
    ifelse(is.null(cv_window),"expanding",as.character(cv_window)),
    cv_skip))
  if(is.null(cv_initial)) cv_initial <- max(lags+horizon+5L, as.integer(nrow(data)*0.5))

  resamples <- .make_resamples(data, cv_initial, horizon, cv_skip, cv_window)
  n_splits  <- length(resamples$splits)
  cat(sprintf("  Splits: %d\n", n_splits))

  strats <- if(arch_l=="nbeats") "nbeats"
            else if(strategy=="both") c("recursive")  # direct is slow per-fold
            else strategy

  oos_all <- list(); metrics_all <- list()

  for(strat in strats){
    cat(sprintf("  Resampling [%s - %s]…\n", toupper(arch_l), strat))
    fc_fn <- .deep_forecastfn(hp, lags, arch_l, epochs, patience, val_fraction,
                               lr_factor, lr_patience, grad_clip, scale, device, strat)

    # Iterate over splits (parallel via furrr)
    process_split <- function(i){
      spl  <- resamples$splits[[i]]
      a_df <- rsample::analysis(spl)
      s_df <- rsample::assessment(spl)
      tryCatch(fc_fn(a_df, s_df), error=function(e) NULL)
    }

    split_results <- if(use_parallel && .is_parallel())
      furrr::future_map(seq_len(n_splits), process_split,
        .options=furrr::furrr_options(seed=TRUE, globals=FALSE), .progress=TRUE)
    else
      lapply(seq_len(n_splits), function(i){
        cat(sprintf("  split %d/%d\r",i,n_splits)); process_split(i)})

    oos_strat <- dplyr::bind_rows(
      Filter(Negate(is.null), split_results)) %>%
      mutate(fold = rep(seq_along(split_results), vapply(split_results, function(r) if(is.null(r)) 0L else nrow(r), integer(1L))))

    if(nrow(oos_strat)==0L){ warning(sprintf("No results for strategy %s",strat)); next }

    # ── Per-fold and overall metrics ──────────────────────────────
    oos_metrics <- oos_strat %>%
      group_by(fold, strategy) %>%
      summarise(
        RMSE  = sqrt(mean(residual^2, na.rm=TRUE)),
        MAE   = mean(abs(residual),   na.rm=TRUE),
        MAPE  = mean(abs(residual/actual)*100, na.rm=TRUE),
        SMAPE = mean(200*abs(residual)/(abs(actual)+abs(predicted)), na.rm=TRUE),
        MBE   = mean(residual, na.rm=TRUE),
        n     = dplyr::n(),
        .groups="drop")

    overall <- oos_metrics %>%
      group_by(strategy) %>%
      summarise(across(c(RMSE,MAE,MAPE,SMAPE,MBE), ~mean(.x,na.rm=TRUE)), .groups="drop")

    cat(sprintf("\n── [%s | %s] Resample CV Overall ──\n", toupper(arch_l), strat))
    print(overall)

    oos_all[[strat]]     <- oos_strat
    metrics_all[[strat]] <- list(by_fold=oos_metrics, overall=overall)
  }

  # Combined OOS tibble (all strategies)
  oos_combined <- dplyr::bind_rows(oos_all)
  cv_metrics   <- dplyr::bind_rows(lapply(metrics_all, `[[`, "overall"))

  list(
    cv_metrics   = cv_metrics,
    by_fold      = dplyr::bind_rows(lapply(metrics_all, `[[`, "by_fold")),
    oos_tbl      = oos_combined,
    resamples    = resamples,
    elapsed_sec  = .toc(t0)
  )
}

# ── CV for statistical models (ARIMA, ETS, ARFIMA, HAR, GARCH) ─
# Wraps forecast/rugarch fitting inside the same split loop.
# Returns the same structure as .resample_cv_deep so downstream
# code is unified.

.resample_cv_stat <- function(data, horizon, cv_initial, cv_window, cv_skip,
                               fit_fn, model_name, use_parallel=FALSE){
  # fit_fn(analysis_df) → list(model=<fitted_obj>, forecast_fn=function(h)->numeric(h))
  t0 <- .tic()
  if(is.null(cv_initial)) cv_initial <- max(as.integer(nrow(data)*0.5), 30L)
  resamples <- .make_resamples(data, cv_initial, horizon, cv_skip, cv_window)
  n_splits  <- length(resamples$splits)
  cat(sprintf("\n── modeltime.resample [%s] | %d splits\n", model_name, n_splits))

  process_split <- function(i){
    spl  <- resamples$splits[[i]]
    a_df <- rsample::analysis(spl)
    s_df <- rsample::assessment(spl)
    n_tst <- nrow(s_df)
    tryCatch({
      fit_out <- fit_fn(a_df)
      fc_vals <- fit_out$forecast_fn(n_tst)
      tibble(fold=i, date=s_df$date, actual=s_df$value, predicted=fc_vals,
             residual=s_df$value-fc_vals, strategy=model_name)
    }, error=function(e){
      warning(sprintf("  [%s] split %d failed: %s", model_name, i, e$message))
      NULL
    })
  }

  split_results <- if(use_parallel && .is_parallel())
    furrr::future_map(seq_len(n_splits), process_split,
      .options=furrr::furrr_options(seed=TRUE, globals=FALSE), .progress=TRUE)
  else
    lapply(seq_len(n_splits), function(i){
      cat(sprintf("  split %d/%d\r",i,n_splits)); process_split(i)})

  oos <- dplyr::bind_rows(Filter(Negate(is.null), split_results))
  if(nrow(oos)==0L){warning(sprintf("No CV results for %s",model_name));return(NULL)}

  overall <- oos %>%
    summarise(
      RMSE  = sqrt(mean(residual^2, na.rm=TRUE)),
      MAE   = mean(abs(residual),   na.rm=TRUE),
      MAPE  = mean(abs(residual/actual)*100, na.rm=TRUE),
      SMAPE = mean(200*abs(residual)/(abs(actual)+abs(predicted)), na.rm=TRUE),
      MBE   = mean(residual, na.rm=TRUE))

  cat(sprintf("\n── [%s] Resample CV Overall ──\n", model_name)); print(overall)

  list(cv_metrics=overall, by_fold=NULL, oos_tbl=oos, resamples=resamples,
       elapsed_sec=.toc(t0))
}

# ── resample_accuracy() thin wrapper (mirrors modeltime API) ────
#' Extract accuracy metrics from a tf_calibrated or cv_results object
#' @export
resample_accuracy <- function(object, ...){
  if(inherits(object,"tf_calibrated")) return(tf_accuracy(object))
  if(is.list(object)&&!is.null(object$cv_metrics)) return(object$cv_metrics)
  stop("Provide tf_calibrated or cv_results list with $cv_metrics.")
}

# ================================================================
# ██  PART 10 — HOLD-OUT OOS EVALUATION  (unchanged logic)
# ================================================================

.holdout_eval <- function(data, lags, horizon, hp, arch, holdout_frac,
                           strategy, epochs, patience, val_fraction,
                           lr_factor, lr_patience, grad_clip, scale, device, verbose){
  t0 <- .tic(); arch_l <- tolower(arch)
  n_total <- nrow(data); n_ho <- max(horizon, floor(n_total*holdout_frac))
  n_train <- n_total - n_ho
  if(n_train < lags+horizon+5L) stop("Hold-out too large. Reduce cv_holdout_frac.")
  tr_df  <- data[seq_len(n_train),]; tst_df <- data[(n_train+1L):n_total,]
  cat(sprintf("\n── Hold-out [%s] | train=%d | test=%d\n",toupper(arch_l),n_train,nrow(tst_df)))
  sc     <- make_scaler(tr_df$value)
  tr_sc  <- if(scale) scale_x(tr_df$value,sc) else tr_df$value
  tst_sc <- if(scale) scale_x(tst_df$value,sc) else tst_df$value
  sv     <- make_supervised(tr_sc, lags)
  n      <- nrow(sv$X); nv <- floor(n*val_fraction); nt <- n-nv
  results <- list()

  .train_tr <- function() train_deep_model(
    sv$X[1:nt,,drop=FALSE], sv$y[1:nt],
    if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
    if(nv>0) sv$y[(nt+1):n] else NULL,
    lags=lags, arch=arch_l, hidden_size=hp$hidden_size, num_layers=hp$num_layers,
    dropout=hp$dropout, dense_units=hp$dense_units, lr=hp$lr, epochs=epochs,
    batch_size=hp$batch_size, patience=patience, lr_factor=lr_factor,
    lr_patience=lr_patience, grad_clip=grad_clip, horizon=1L, verbose=verbose, device=device)

  if(strategy %in% c("recursive","both")){
    res <- .train_tr(); fv <- .rolling_oos(res$model, tr_sc, tst_sc, lags, sc, device)
    m   <- .safe_metrics(tst_df$value, fv)
    cat(sprintf("  [Recursive] RMSE=%.4f|MAE=%.4f|MAPE=%.2f%%\n",m$RMSE,m$MAE,m$MAPE%||%NA))
    results$recursive <- tibble(fold=1L, h=seq_len(nrow(tst_df)), date=tst_df$date,
      actual=tst_df$value, fitted=fv, residual=tst_df$value-fv, strategy="recursive")
    results$metrics_recursive <- m; results$model_recursive <- res$model
    results$history_recursive <- res$history
  }
  if(strategy %in% c("direct","both")){
    sv1 <- make_supervised_direct(tr_sc, lags, 1L); n1 <- nrow(sv1$X)
    nv1 <- floor(n1*val_fraction); nt1 <- n1-nv1
    r1  <- train_deep_model(sv1$X[1:nt1,,drop=FALSE], sv1$y[1:nt1],
             if(nv1>0) sv1$X[(nt1+1):n1,,drop=FALSE] else NULL,
             if(nv1>0) sv1$y[(nt1+1):n1] else NULL,
             lags=lags, arch=arch_l, hidden_size=hp$hidden_size, num_layers=hp$num_layers,
             dropout=hp$dropout, dense_units=hp$dense_units, lr=hp$lr, epochs=epochs,
             batch_size=hp$batch_size, patience=patience, lr_factor=lr_factor,
             lr_patience=lr_patience, grad_clip=grad_clip, horizon=1L, verbose=0L, device=device)
    fv_d <- .rolling_oos(r1$model, tr_sc, tst_sc, lags, sc, device)
    m_d  <- .safe_metrics(tst_df$value, fv_d)
    cat(sprintf("  [Direct]    RMSE=%.4f|MAE=%.4f|MAPE=%.2f%%\n",m_d$RMSE,m_d$MAE,m_d$MAPE%||%NA))
    results$direct <- tibble(fold=1L, h=seq_len(nrow(tst_df)), date=tst_df$date,
      actual=tst_df$value, fitted=fv_d, residual=tst_df$value-fv_d, strategy="direct")
    results$metrics_direct <- m_d
  }
  results$elapsed_sec <- .toc(t0); results$n_train <- n_train
  results$n_test <- nrow(tst_df); results$train_df <- tr_df; results$test_df <- tst_df
  results
}


# ================================================================
# ██  PART 11 — MASTER: auto_deep() + wrappers
# ================================================================

#' @export
auto_deep <- function(data, date_col="date", value_col="value", arch="lstm",
                       horizon=12L, strategy=c("both","recursive","direct"),
                       device=NULL, config=NULL,
                       run_bo=NULL, bo_init=NULL, bo_iter=NULL,
                       bo_epochs=NULL, bo_patience=NULL, bo_bounds=NULL,
                       final_epochs=NULL, final_patience=NULL,
                       run_cv=NULL, cv_initial=NULL, cv_window=NULL, cv_skip=NULL,
                       cv_holdout_frac=NULL, val_fraction=NULL,
                       lags=NULL, scale=NULL, seed=NULL, verbose=NULL, use_parallel=NULL){
  total_t0 <- .tic()
  cfg <- .resolve_cfg(config, list(
    run_bo=run_bo, bo_init=bo_init, bo_iter=bo_iter, bo_epochs=bo_epochs,
    bo_patience=bo_patience, bo_bounds=bo_bounds, final_epochs=final_epochs,
    final_patience=final_patience, run_cv=run_cv, cv_initial=cv_initial,
    cv_window=cv_window, cv_skip=cv_skip, cv_holdout_frac=cv_holdout_frac,
    val_fraction=val_fraction, lags=lags, scale=scale, seed=seed, verbose=verbose))
  set.seed(cfg$seed); torch_manual_seed(cfg$seed)
  arch_l   <- .check_arch(arch); strategy <- match.arg(strategy)
  if(is.null(device)) device <- get_device()
  use_par  <- (use_parallel%||%TRUE) && .is_parallel() && device=="cpu"
  if(is.null(cfg$lags)) cfg$lags <- max(12L, as.integer(ceiling(horizon*1.5)))
  lags_v   <- cfg$lags

  stopifnot(is.data.frame(data), date_col %in% names(data), value_col %in% names(data))
  df <- data %>% rename(date=!!sym(date_col), value=!!sym(value_col)) %>%
    arrange(date) %>% select(date, value)
  if(anyNA(df$value)){warning("Missing→interp."); df$value <- zoo::na.approx(df$value,na.rm=FALSE)}

  cat(sprintf("\nAuto%s | n=%d | h=%d | lags=%d | strategy=%s | device=%s | par=%s\n",
    toupper(arch_l), nrow(df), horizon, lags_v, strategy, device, use_par))
  cat(sprintf("  BO=%s [ParBayesianOptimization] | CV=%s%s\n",
    if(cfg$run_bo)"ON" else "OFF",
    if(cfg$run_cv) sprintf("modeltime.resample (skip=%d)",cfg$cv_skip)
    else sprintf("hold-out %.0f%%",cfg$cv_holdout_frac*100),
    if(cfg$run_cv&&!is.null(cfg$cv_window)) sprintf(" [window=%d]",cfg$cv_window) else ""))

  sc      <- make_scaler(df$value)
  scaled  <- if(cfg$scale) scale_x(df$value,sc) else df$value
  def_hp  <- list(hidden_size=cfg$hidden_size, num_layers=cfg$num_layers,
                  dropout=cfg$dropout, dense_units=cfg$dense_units,
                  lr=cfg$lr, batch_size=cfg$batch_size)
  runtime <- list()

  # ── Bayesian Optimisation (ParBayesianOptimization) ──────────
  bo_result <- NULL
  if(cfg$run_bo){
    bounds  <- .resolve_bounds(cfg$bo_bounds)
    bo_out  <- .bo_optimize(scaled, lags_v, arch_l, cfg$val_fraction,
                             cfg$bo_iter, cfg$bo_init, cfg$bo_epochs, cfg$bo_patience,
                             cfg$bo_acq, cfg$bo_kappa, bounds, device, cfg$seed,
                             horizon=horizon, use_parallel=use_par)
    best_hp <- bo_out$best_params; bo_result <- bo_out$bo_result
    runtime$bo_sec <- bo_out$elapsed_sec
  } else { best_hp <- def_hp; runtime$bo_sec <- 0 }

  # ── CV (modeltime.resample) or hold-out ──────────────────────
  cv_results <- NULL; holdout_results <- NULL
  if(cfg$run_cv){
    cv_out <- .resample_cv_deep(df, lags_v, horizon, best_hp, arch_l,
                                 cfg$cv_initial, cfg$cv_window, cfg$cv_skip, strategy,
                                 cfg$final_epochs, cfg$final_patience, cfg$val_fraction,
                                 cfg$lr_factor, cfg$lr_patience, cfg$grad_clip,
                                 cfg$scale, device, 0L, use_parallel=use_par)
    cv_results <- cv_out; runtime$cv_sec <- cv_out$elapsed_sec
  } else {
    ho_out <- .holdout_eval(df, lags_v, horizon, best_hp, arch_l, cfg$cv_holdout_frac,
                             strategy, cfg$final_epochs, cfg$final_patience, cfg$val_fraction,
                             cfg$lr_factor, cfg$lr_patience, cfg$grad_clip, cfg$scale,
                             device, cfg$verbose)
    holdout_results <- ho_out; runtime$holdout_sec <- ho_out$elapsed_sec
  }

  # ── Final model on full series ────────────────────────────────
  cat(sprintf("\n── Final [%s] on full series ──\n", toupper(arch_l)))
  ft0     <- .tic()
  sv_full <- make_supervised(scaled, lags_v)
  n       <- nrow(sv_full$X); nv <- floor(n*cfg$val_fraction); nt <- n-nv
  # Use matrixStats for fast column scaling of the supervised matrix
  sc_m    <- scale_matrix_cols(sv_full$X)
  X_all_sc <- sc_m$X_scaled
  X_tr    <- X_all_sc[1:nt,,drop=FALSE]; y_tr <- sv_full$y[1:nt]
  X_vl    <- if(nv>0) X_all_sc[(nt+1):n,,drop=FALSE] else NULL
  y_vl    <- if(nv>0) sv_full$y[(nt+1):n] else NULL
  last_win <- tail(scaled, lags_v)
  freq     <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by=freq, length.out=horizon+1L)[-1L]

  .ci <- function(mdl, Xm, yv, fc){
    is_nb  <- arch_l=="nbeats"; p <- predict_deep(mdl, Xm, device, is_nb)
    res_sd <- sd(unscale_x(yv,sc) - unscale_x(p,sc))
    ci_w   <- qnorm(0.975)*res_sd*sqrt(seq_len(horizon))
    list(lower_95=fc-ci_w, upper_95=fc+ci_w)
  }

  model_recursive <- NULL; history_recursive <- NULL
  forecast_rec_df <- NULL; direct_models <- NULL; forecast_dir_df <- NULL

  if(arch_l=="nbeats"){
    cat("  [N-BEATS] Training…\n")
    res_nb <- train_deep_model(X_tr, y_tr, X_vl, y_vl, lags=lags_v, arch="nbeats",
      hidden_size=best_hp$hidden_size, num_layers=best_hp$num_layers, dropout=best_hp$dropout,
      dense_units=best_hp$dense_units, lr=best_hp$lr, epochs=cfg$final_epochs,
      batch_size=best_hp$batch_size, patience=cfg$final_patience, lr_factor=cfg$lr_factor,
      grad_clip=cfg$grad_clip, horizon=horizon, verbose=cfg$verbose, device=device)
    model_recursive <- res_nb$model; history_recursive <- res_nb$history
    fc_nb <- nbeats_forecast(model_recursive, last_win, horizon, sc, device)
    ci_nb <- .ci(model_recursive, X_tr, y_tr, fc_nb)
    forecast_rec_df <- tibble(date=fut_dates, forecast=fc_nb,
      lower_95=ci_nb$lower_95, upper_95=ci_nb$upper_95, strategy="nbeats")
    cat("  [N-BEATS] Done.\n")
  } else {
    if(strategy %in% c("recursive","both")){
      cat("  [Recursive] Training…\n")
      res <- train_deep_model(X_tr, y_tr, X_vl, y_vl, lags=lags_v, arch=arch_l,
        hidden_size=best_hp$hidden_size, num_layers=best_hp$num_layers, dropout=best_hp$dropout,
        dense_units=best_hp$dense_units, lr=best_hp$lr, epochs=cfg$final_epochs,
        batch_size=best_hp$batch_size, patience=cfg$final_patience, lr_factor=cfg$lr_factor,
        lr_patience=cfg$lr_patience, grad_clip=cfg$grad_clip, horizon=1L,
        verbose=cfg$verbose, device=device)
      model_recursive <- res$model; history_recursive <- res$history
      fc  <- recursive_forecast(model_recursive, last_win, horizon, sc, device)
      ci  <- .ci(model_recursive, X_tr, y_tr, fc)
      forecast_rec_df <- tibble(date=fut_dates, forecast=fc,
        lower_95=ci$lower_95, upper_95=ci$upper_95, strategy="recursive")
      cat("  [Recursive] Done.\n")
    }
    if(strategy %in% c("direct","both")){
      cat("  [Direct] Training H models…\n")
      direct_models <- train_direct_models(scaled, lags_v, horizon, best_hp, arch_l,
        cfg$final_epochs, cfg$final_patience, cfg$val_fraction, cfg$lr_factor,
        cfg$lr_patience, cfg$grad_clip, device, 0L, use_parallel=use_par)
      fc  <- direct_forecast(direct_models, last_win, sc, device)
      sv1 <- make_supervised_direct(scaled, lags_v, 1L)
      nt1 <- nrow(sv1$X) - floor(nrow(sv1$X)*cfg$val_fraction)
      p1  <- predict_deep(direct_models[[1]], sv1$X[1:nt1,,drop=FALSE], device)
      res_sd1 <- sd(unscale_x(sv1$y[1:nt1],sc) - unscale_x(p1,sc))
      ci_w1   <- qnorm(0.975)*res_sd1*sqrt(seq_len(horizon))
      forecast_dir_df <- tibble(date=fut_dates, forecast=fc,
        lower_95=fc-ci_w1, upper_95=fc+ci_w1, strategy="direct")
      cat("  [Direct] Done.\n")
    }
  }
  runtime$final_train_sec <- .toc(ft0); runtime$total_sec <- .toc(total_t0)
  cat(sprintf("\n✓ Auto%s | total=%s | BO=%s | eval=%s | final=%s\n",
    toupper(arch_l), .fmt_elapsed(runtime$total_sec), .fmt_elapsed(runtime$bo_sec%||%0),
    .fmt_elapsed((runtime$cv_sec%||%runtime$holdout_sec)%||%0),
    .fmt_elapsed(runtime$final_train_sec)))

  structure(list(
    model_type=arch_l, strategy=strategy, horizon=horizon, lags=lags_v,
    forecast_recursive=forecast_rec_df, forecast_direct=forecast_dir_df,
    best_params=best_hp, bo_result=bo_result,
    cv_results=cv_results, holdout_results=holdout_results,
    model_recursive=model_recursive, direct_models=direct_models,
    history_recursive=history_recursive,
    scaler=sc, data=df, config_used=cfg, runtime=runtime
  ), class=c(paste0("auto_",arch_l),"auto_rnn"))
}

#' @export
auto_rnn_torch <- function(data,...,arch="lstm") auto_deep(data,...,arch=arch)
#' @export
auto_lstm   <- function(data,...) auto_deep(data,...,arch="lstm")
#' @export
auto_gru    <- function(data,...) auto_deep(data,...,arch="gru")
#' @export
auto_rnn    <- function(data,...) auto_deep(data,...,arch="rnn")
#' @export
auto_nbeats <- function(data,...) auto_deep(data,...,arch="nbeats")


# ================================================================
# ██  PART 12 — AUTO-ARIMA  (modeltime.resample CV)
# ================================================================

#' @export
auto_arima_ts <- function(data, date_col="date", value_col="value",
                           horizon=12L, level=c(80,95), stepwise=TRUE,
                           approximation=NULL, lambda=NULL,
                           run_cv=TRUE, cv_initial=NULL, cv_window=NULL,
                           cv_skip=1L, cv_holdout_frac=0.20,
                           xreg=NULL, xreg_future=NULL, seed=42L, verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  df <- data %>% rename(date=!!sym(date_col), value=!!sym(value_col)) %>%
    arrange(date) %>% select(date, value)
  if(anyNA(df$value)){df$value <- zoo::na.approx(df$value,na.rm=FALSE)}
  ts_obj <- .df_to_ts(df); freq <- .ts_freq(df$date)
  if(verbose>0L) cat("\n── Auto-ARIMA ─────────────────────────────────────────────\n")
  fit_t0 <- .tic()
  model  <- forecast::auto.arima(ts_obj, xreg=xreg, stepwise=stepwise,
    approximation=approximation, lambda=lambda, ic="aicc", trace=verbose>0L)
  fit_sec <- .toc(fit_t0)
  if(verbose>0L){
    cat(sprintf("\n  ARIMA(%d,%d,%d)",model$arma[1],model$arma[6],model$arma[2]))
    if(freq>1L) cat(sprintf("(%d,%d,%d)[%d]",model$arma[3],model$arma[7],model$arma[4],freq))
    cat(sprintf(" | AICc=%.2f\n",model$aicc))}
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by=freq_str, length.out=horizon+1L)[-1L]
  fc_obj    <- forecast::forecast(model, h=horizon, level=level, xreg=xreg_future)
  forecast_df <- tibble(date=fut_dates, forecast=as.numeric(fc_obj$mean),
    lower_80=as.numeric(fc_obj$lower[,1]), upper_80=as.numeric(fc_obj$upper[,1]),
    lower_95=if(length(level)>=2L) as.numeric(fc_obj$lower[,2]) else NA_real_,
    upper_95=if(length(level)>=2L) as.numeric(fc_obj$upper[,2]) else NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0
  if(run_cv){
    ev_t0 <- .tic()
    fit_fn <- local({sw_=stepwise; ap_=approximation; lm_=lambda; fr_=freq
      function(a_df){
        ts_a <- ts(a_df$value, frequency=fr_,
                   start=c(as.integer(format(min(a_df$date),"%Y")),
                           if(fr_==12L) as.integer(format(min(a_df$date),"%m")) else 1L))
        m_i <- tryCatch(forecast::auto.arima(ts_a,stepwise=sw_,approximation=ap_,
                          lambda=lm_,ic="aicc"), error=function(e) NULL)
        list(model=m_i, forecast_fn=function(h){
          if(is.null(m_i)) return(rep(NA_real_,h))
          as.numeric(forecast::forecast(m_i,h=h)$mean)
        })
      }})
    cv_out <- .resample_cv_stat(df, horizon, cv_initial, cv_window, cv_skip, fit_fn, "arima")
    if(!is.null(cv_out)) cv_results <- cv_out
    if(verbose>0L&&!is.null(cv_results)){cat("\n── ARIMA resample CV ──\n");print(cv_results$cv_metrics)}
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic(); n_total <- nrow(df)
    n_ho <- max(horizon, floor(n_total*cv_holdout_frac))
    n_tr <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    cat(sprintf("\n── ARIMA hold-out (train=%d,test=%d) ──\n",n_tr,nrow(tst_df)))
    ts_tr <- ts(tr_df$value, frequency=freq,
                start=c(as.integer(format(min(tr_df$date),"%Y")),
                        if(freq==12L) as.integer(format(min(tr_df$date),"%m")) else 1L))
    m_ho <- tryCatch(forecast::auto.arima(ts_tr,stepwise=stepwise,ic="aicc"),error=function(e) NULL)
    n_tst <- nrow(tst_df); fv <- numeric(n_tst)
    if(!is.null(m_ho)){
      all_ts <- ts(df$value, frequency=freq,
                   start=c(as.integer(format(min(df$date),"%Y")),
                           if(freq==12L) as.integer(format(min(df$date),"%m")) else 1L))
      refit  <- tryCatch(forecast::Arima(all_ts,model=m_ho),error=function(e) NULL)
      if(!is.null(refit)){af<-as.numeric(fitted(refit));fv<-af[(n_tr+1L):(n_tr+n_tst)]}
      else fv <- tryCatch(as.numeric(forecast::forecast(m_ho,h=n_tst)$mean),
                           error=function(e) rep(NA_real_,n_tst))
    } else fv <- rep(NA_real_,n_tst)
    m_ho <- .safe_metrics(tst_df$value, fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f\n",m_ho$RMSE,m_ho$MAE))
    holdout_results <- tibble(fold=1L, h=seq_len(n_tst), date=tst_df$date,
      actual=tst_df$value, fitted=fv, residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec, eval_sec=eval_sec, total_sec=.toc(total_t0))
  cat(sprintf("\n✓ Auto-ARIMA | total=%s\n",.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model, forecast=forecast_df, cv_results=cv_results,
    holdout_results=holdout_results, data=df, horizon=horizon, level=level,
    ts_obj=ts_obj, freq=freq, runtime=runtime), class="auto_arima_ts")
}


# ================================================================
# ██  PART 13 — AUTO-ETS  (modeltime.resample CV)
# ================================================================

#' @export
auto_ets_ts <- function(data, date_col="date", value_col="value",
                         horizon=12L, level=c(80,95), model_spec="ZZZ", damped=NULL,
                         run_cv=TRUE, cv_initial=NULL, cv_window=NULL,
                         cv_skip=1L, cv_holdout_frac=0.20, seed=42L, verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  df <- data %>% rename(date=!!sym(date_col), value=!!sym(value_col)) %>%
    arrange(date) %>% select(date, value)
  if(anyNA(df$value)) df$value <- zoo::na.approx(df$value, na.rm=FALSE)
  ts_obj <- .df_to_ts(df); freq <- .ts_freq(df$date)
  if(verbose>0L) cat("\n── Auto-ETS ──────────────────────────────────────────────\n")
  fit_t0 <- .tic()
  model  <- forecast::ets(ts_obj, model=model_spec, damped=damped, ic="aicc")
  fit_sec <- .toc(fit_t0)
  if(verbose>0L) cat(sprintf("  ETS(%s) | AICc=%.2f\n",model$method,model$aicc))
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by=freq_str, length.out=horizon+1L)[-1L]
  fc_obj    <- forecast::forecast(model, h=horizon, level=level)
  forecast_df <- tibble(date=fut_dates, forecast=as.numeric(fc_obj$mean),
    lower_80=as.numeric(fc_obj$lower[,1]), upper_80=as.numeric(fc_obj$upper[,1]),
    lower_95=if(length(level)>=2L) as.numeric(fc_obj$lower[,2]) else NA_real_,
    upper_95=if(length(level)>=2L) as.numeric(fc_obj$upper[,2]) else NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0
  if(run_cv){
    ev_t0 <- .tic()
    fit_fn <- local({ms_=model_spec; dm_=damped; fr_=freq
      function(a_df){
        ts_a <- ts(a_df$value, frequency=fr_,
                   start=c(as.integer(format(min(a_df$date),"%Y")),
                           if(fr_==12L) as.integer(format(min(a_df$date),"%m")) else 1L))
        m_i <- tryCatch(forecast::ets(ts_a,model=ms_,damped=dm_,ic="aicc"),error=function(e) NULL)
        list(model=m_i, forecast_fn=function(h){
          if(is.null(m_i)) return(rep(NA_real_,h))
          as.numeric(forecast::forecast(m_i,h=h)$mean)
        })
      }})
    cv_out <- .resample_cv_stat(df, horizon, cv_initial, cv_window, cv_skip, fit_fn, "ets")
    if(!is.null(cv_out)) cv_results <- cv_out
    if(verbose>0L&&!is.null(cv_results)){cat("\n── ETS resample CV ──\n");print(cv_results$cv_metrics)}
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic(); n_total <- nrow(df)
    n_ho <- max(horizon, floor(n_total*cv_holdout_frac))
    n_tr <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    cat(sprintf("\n── ETS hold-out (train=%d,test=%d) ──\n",n_tr,nrow(tst_df)))
    ts_tr <- ts(tr_df$value, frequency=freq,
                start=c(as.integer(format(min(tr_df$date),"%Y")),
                        if(freq==12L) as.integer(format(min(tr_df$date),"%m")) else 1L))
    m_ho <- tryCatch(forecast::ets(ts_tr,model=model_spec,ic="aicc"),error=function(e) NULL)
    n_tst <- nrow(tst_df)
    fv    <- if(!is.null(m_ho)) tryCatch(as.numeric(forecast::forecast(m_ho,h=n_tst)$mean),
               error=function(e) rep(NA_real_,n_tst)) else rep(NA_real_,n_tst)
    m_ho  <- .safe_metrics(tst_df$value, fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f\n",m_ho$RMSE,m_ho$MAE))
    holdout_results <- tibble(fold=1L, h=seq_len(n_tst), date=tst_df$date,
      actual=tst_df$value, fitted=fv, residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec, eval_sec=eval_sec, total_sec=.toc(total_t0))
  cat(sprintf("\n✓ Auto-ETS | method=%s | total=%s\n",model$method,.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model, forecast=forecast_df, cv_results=cv_results,
    holdout_results=holdout_results, data=df, horizon=horizon, level=level,
    ts_obj=ts_obj, freq=freq, runtime=runtime), class="auto_ets_ts")
}


# ================================================================
# ██  PART 14 — AUTO-ARFIMA, HAR-RV, GARCH (resample CV)
# ================================================================

#' @export
auto_arfima_ts <- function(data, date_col="date", value_col="value",
                            horizon=12L, level=c(80,95), drange=c(0,0.5),
                            ar_max=5L, ma_max=5L, run_cv=TRUE,
                            cv_initial=NULL, cv_window=NULL, cv_skip=1L,
                            cv_holdout_frac=0.20, seed=42L, verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  df <- data %>% rename(date=!!sym(date_col), value=!!sym(value_col)) %>%
    arrange(date) %>% select(date, value)
  if(anyNA(df$value)) df$value <- zoo::na.approx(df$value, na.rm=FALSE)
  ts_obj <- .df_to_ts(df); freq <- .ts_freq(df$date)
  fit_t0 <- .tic()
  fd_fit <- fracdiff::fracdiff(ts_obj, drange=drange, ar=ar_max, ma=ma_max); d_hat <- fd_fit$d
  if(verbose>0L) cat(sprintf("\n── ARFIMA | d=%.6f\n",d_hat))
  model  <- tryCatch(forecast::arfima(ts_obj, drange=drange, estim="mle"),
                     error=function(e){warning("arfima() fallback."); fd_fit})
  fit_sec <- .toc(fit_t0)
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by=freq_str, length.out=horizon+1L)[-1L]
  fc_obj <- tryCatch(forecast::forecast(model, h=horizon, level=level), error=function(e) NULL)
  forecast_df <- if(!is.null(fc_obj))
    tibble(date=fut_dates, forecast=as.numeric(fc_obj$mean),
      lower_80=as.numeric(fc_obj$lower[,1]), upper_80=as.numeric(fc_obj$upper[,1]),
      lower_95=if(length(level)>=2L) as.numeric(fc_obj$lower[,2]) else NA_real_,
      upper_95=if(length(level)>=2L) as.numeric(fc_obj$upper[,2]) else NA_real_)
  else tibble(date=fut_dates, forecast=rep(NA_real_,horizon),
    lower_80=NA_real_, upper_80=NA_real_, lower_95=NA_real_, upper_95=NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0
  if(run_cv){
    ev_t0 <- .tic()
    fit_fn <- local({dr_=drange; fr_=freq
      function(a_df){
        ts_a <- ts(a_df$value, frequency=fr_,
                   start=c(as.integer(format(min(a_df$date),"%Y")),
                           if(fr_==12L) as.integer(format(min(a_df$date),"%m")) else 1L))
        m_i <- tryCatch(forecast::arfima(ts_a, drange=dr_, estim="mle"), error=function(e) NULL)
        list(model=m_i, forecast_fn=function(h){
          if(is.null(m_i)) return(rep(NA_real_,h))
          tryCatch(as.numeric(forecast::forecast(m_i,h=h)$mean),
                   error=function(e) rep(NA_real_,h))
        })
      }})
    cv_out <- .resample_cv_stat(df, horizon, cv_initial, cv_window, cv_skip, fit_fn, "arfima")
    if(!is.null(cv_out)) cv_results <- cv_out
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic(); n_total <- nrow(df)
    n_ho <- max(horizon, floor(n_total*cv_holdout_frac))
    n_tr <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    ts_tr <- ts(tr_df$value, frequency=freq,
                start=c(as.integer(format(min(tr_df$date),"%Y")),
                        if(freq==12L) as.integer(format(min(tr_df$date),"%m")) else 1L))
    m_ho <- tryCatch(forecast::arfima(ts_tr,drange=drange,estim="mle"),error=function(e) NULL)
    n_tst <- nrow(tst_df); fv <- numeric(n_tst)
    if(!is.null(m_ho)){
      all_ts <- ts(df$value, frequency=freq,
                   start=c(as.integer(format(min(df$date),"%Y")),
                           if(freq==12L) as.integer(format(min(df$date),"%m")) else 1L))
      refit <- tryCatch(forecast::Arima(all_ts,model=m_ho),error=function(e) NULL)
      if(!is.null(refit)){af<-as.numeric(fitted(refit));fv<-af[(n_tr+1L):(n_tr+n_tst)]}
      else fv <- tryCatch(as.numeric(forecast::forecast(m_ho,h=n_tst)$mean),
                           error=function(e) rep(NA_real_,n_tst))
    } else fv <- rep(NA_real_,n_tst)
    m_ho <- .safe_metrics(tst_df$value, fv)
    holdout_results <- tibble(fold=1L, h=seq_len(n_tst), date=tst_df$date,
      actual=tst_df$value, fitted=fv, residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec, eval_sec=eval_sec, total_sec=.toc(total_t0))
  cat(sprintf("\n✓ ARFIMA | d=%.4f | total=%s\n",d_hat,.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model, fd_fit=fd_fit, d_hat=d_hat, forecast=forecast_df,
    cv_results=cv_results, holdout_results=holdout_results,
    data=df, horizon=horizon, level=level, ts_obj=ts_obj, freq=freq, runtime=runtime),
    class="auto_arfima_ts")
}


#' @export
auto_har_rv <- function(data, date_col="date", rv_col="value", horizon=1L,
                         variant=c("standard","harj","harq","harcj"),
                         d_lags=1L, w_lags=5L, m_lags=22L,
                         jump_col=NULL, rq_col=NULL, log_transform=FALSE,
                         run_cv=TRUE, cv_initial=NULL, cv_window=NULL, cv_skip=1L,
                         cv_holdout_frac=0.20, seed=42L, verbose=1L){
  total_t0 <- .tic(); set.seed(seed); variant <- match.arg(variant)
  df <- data %>% rename(date=!!sym(date_col), value=!!sym(rv_col)) %>%
    arrange(date) %>% select(date, value, any_of(c(jump_col,rq_col)))
  if(anyNA(df$value)) df$value <- zoo::na.approx(df$value, na.rm=FALSE)
  rv_use <- if(log_transform) log(pmax(df$value,1e-10)) else df$value
  jmp    <- if(!is.null(jump_col)&&jump_col%in%names(df)) df[[jump_col]] else NULL
  rq     <- if(!is.null(rq_col)  &&rq_col  %in%names(df)) df[[rq_col]]   else NULL

  # ── HAR feature builder: uses RcppRoll::roll_mean for fast lags ──
  .build_har_feats <- function(rv){
    n    <- length(rv)
    lag1 <- c(rep(NA_real_,d_lags), rv[seq_len(n-d_lags)])
    # RcppRoll::roll_mean replaces zoo::rollmean loops
    lagw <- .roll_mean_right(rv, w_lags)
    lagw <- c(rep(NA_real_, w_lags), lagw[seq_len(n-w_lags)])
    lagm <- .roll_mean_right(rv, m_lags)
    lagm <- c(rep(NA_real_, m_lags), lagm[seq_len(n-m_lags)])
    X <- data.frame(RV_d=lag1, RV_w=lagw, RV_m=lagm)
    if(!is.null(jmp)&&variant%in%c("harj","harcj")){
      cont   <- pmax(rv-jmp, 0)
      lag1_c <- c(rep(NA_real_,d_lags), cont[seq_len(n-d_lags)])
      lag1_j <- c(rep(NA_real_,d_lags), jmp[seq_len(n-d_lags)])
      if(variant=="harj") X$J_d <- lag1_j
      else {X$RV_d <- NULL; X$C_d <- lag1_c; X$J_d <- lag1_j}
    }
    if(!is.null(rq)&&variant=="harq"){
      X$RQ_d <- c(rep(NA_real_,d_lags), sqrt(rq[seq_len(n-d_lags)]))
    }
    X
  }

  feats  <- .build_har_feats(rv_use)
  har_df <- na.omit(data.frame(y=rv_use, feats, date=df$date))
  fml    <- as.formula(paste("y ~", paste(names(feats),collapse=" + ")))
  fit_t0 <- .tic(); model <- lm(fml, data=har_df); fit_sec <- .toc(fit_t0)
  if(verbose>0L){cat(sprintf("\n── HAR-RV [%s]\n",variant)); print(summary(model))}

  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by=freq_str, length.out=horizon+1L)[-1L]
  .har_fc <- function(mdl, rv_h, h, lt){
    preds <- numeric(h)
    for(step in seq_len(h)){
      n_h  <- length(rv_h); lag1 <- rv_h[n_h]
      lagw <- mean(rv_h[max(1,n_h-w_lags+1):n_h])
      lagm <- mean(rv_h[max(1,n_h-m_lags+1):n_h])
      nd   <- data.frame(RV_d=lag1, RV_w=lagw, RV_m=lagm)
      for(nm in setdiff(names(feats),c("RV_d","RV_w","RV_m","C_d")))
        nd[[nm]] <- mean(feats[[nm]],na.rm=TRUE)
      if("C_d"%in%names(feats)) nd$C_d <- lag1
      p <- predict(mdl, newdata=nd)
      preds[step] <- if(lt) exp(p) else p
      rv_h <- c(rv_h, if(lt) p else preds[step])
    }; preds
  }
  forecast_df <- tibble(date=fut_dates,
    forecast=.har_fc(model, tail(rv_use,max(m_lags,60L)), horizon, log_transform),
    lower_95=NA_real_, upper_95=NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0
  if(run_cv){
    ev_t0 <- .tic()
    fit_fn <- local({fml_=fml; d_=d_lags; w_=w_lags; m_=m_lags; lt_=log_transform
      function(a_df){
        rv_a  <- if(lt_) log(pmax(a_df$value,1e-10)) else a_df$value
        n     <- length(rv_a)
        lag1  <- c(rep(NA_real_,d_), rv_a[seq_len(n-d_)])
        lagw  <- .roll_mean_right(rv_a, w_)
        lagw  <- c(rep(NA_real_,w_), lagw[seq_len(n-w_)])
        lagm  <- .roll_mean_right(rv_a, m_)
        lagm  <- c(rep(NA_real_,m_), lagm[seq_len(n-m_)])
        df_tr <- na.omit(data.frame(y=rv_a, RV_d=lag1, RV_w=lagw, RV_m=lagm))
        m_i   <- tryCatch(lm(y~RV_d+RV_w+RV_m, data=df_tr), error=function(e) NULL)
        list(model=m_i, forecast_fn=function(h){
          if(is.null(m_i)||nrow(df_tr)<5L) return(rep(NA_real_,h))
          rv_h  <- tail(rv_a, max(m_,60L)); preds <- numeric(h)
          for(step in seq_len(h)){
            n_h <- length(rv_h)
            nd  <- data.frame(RV_d=rv_h[n_h],
                              RV_w=mean(rv_h[max(1,n_h-w_+1):n_h]),
                              RV_m=mean(rv_h[max(1,n_h-m_+1):n_h]))
            p   <- predict(m_i, newdata=nd)
            preds[step] <- if(lt_) exp(p) else p
            rv_h <- c(rv_h, if(lt_) p else preds[step])
          }; preds
        })
      }})
    cv_out <- .resample_cv_stat(df, horizon, cv_initial, cv_window, cv_skip, fit_fn, "har")
    if(!is.null(cv_out)) cv_results <- cv_out
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic(); n_total <- nrow(df)
    n_ho <- max(horizon, floor(n_total*cv_holdout_frac))
    n_tr <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    rv_tr <- if(log_transform) log(pmax(tr_df$value,1e-10)) else tr_df$value
    f_tr  <- .build_har_feats(rv_tr); df_tr <- na.omit(data.frame(y=rv_tr, f_tr))
    m_ho  <- tryCatch(lm(fml, data=df_tr), error=function(e) NULL)
    n_tst <- nrow(tst_df)
    fv    <- if(!is.null(m_ho)) .har_fc(m_ho, tail(rv_tr,max(m_lags,60L)), n_tst, log_transform)
             else rep(NA_real_,n_tst)
    holdout_results <- tibble(fold=1L, h=seq_len(n_tst), date=tst_df$date,
      actual=tst_df$value, fitted=fv, residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec, eval_sec=eval_sec, total_sec=.toc(total_t0))
  cat(sprintf("\n✓ HAR-RV | %s | total=%s\n",variant,.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model, variant=variant, forecast=forecast_df, cv_results=cv_results,
    holdout_results=holdout_results, data=df, horizon=horizon, log_transform=log_transform,
    d_lags=d_lags, w_lags=w_lags, m_lags=m_lags, runtime=runtime), class="auto_har_rv")
}


#' @export
auto_garch <- function(data, date_col="date", value_col="value", horizon=10L,
                        model=c("sGARCH","eGARCH","gjrGARCH","iGARCH","csGARCH"),
                        p=1L, q=1L, mean_model=c("ARMA","zero","AR","MA"),
                        arma_order=c(0L,0L), distribution="norm",
                        run_cv=TRUE, cv_initial=NULL, cv_window=NULL, cv_skip=1L,
                        cv_holdout_frac=0.20, seed=42L, verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  model_type <- match.arg(model); mean_m <- match.arg(mean_model)
  df <- data %>% rename(date=!!sym(date_col), value=!!sym(value_col)) %>%
    arrange(date) %>% select(date, value)
  if(anyNA(df$value)) df$value <- zoo::na.approx(df$value, na.rm=FALSE)
  .mk_spec <- function() rugarch::ugarchspec(
    variance.model=list(model=model_type, garchOrder=c(p,q)),
    mean.model=list(armaOrder=arma_order, include.mean=if(mean_m=="zero") FALSE else TRUE),
    distribution.model=distribution)
  spec <- .mk_spec(); fit_t0 <- .tic()
  model_fit <- tryCatch(rugarch::ugarchfit(spec=spec, data=df$value, solver="hybrid"),
    error=function(e) tryCatch(rugarch::ugarchfit(spec=spec, data=df$value, solver="solnp"),
      error=function(e2) stop(sprintf("GARCH fit failed: %s",e2$message))))
  fit_sec <- .toc(fit_t0)
  if(verbose>0L){ic<-rugarch::infocriteria(model_fit)
    cat(sprintf("\n── GARCH[%s(%d,%d)] dist=%s | AIC=%.4f\n",model_type,p,q,distribution,ic[1]))}
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date), by=freq_str, length.out=horizon+1L)[-1L]
  fc_obj    <- rugarch::ugarchforecast(model_fit, n.ahead=horizon)
  sigma_fc  <- as.numeric(rugarch::sigma(fc_obj)); mean_fc <- as.numeric(rugarch::fitted(fc_obj))
  forecast_df <- tibble(date=fut_dates, forecast=mean_fc, sigma=sigma_fc, variance=sigma_fc^2,
    lower_95=mean_fc-1.96*sigma_fc, upper_95=mean_fc+1.96*sigma_fc)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0
  if(run_cv){
    ev_t0 <- .tic()
    fit_fn <- local({mt_=model_type;p_=p;q_=q;ao_=arma_order;dist_=distribution;mm_=mean_m
      function(a_df){
        sp <- rugarch::ugarchspec(
          variance.model=list(model=mt_,garchOrder=c(p_,q_)),
          mean.model=list(armaOrder=ao_,include.mean=if(mm_=="zero") FALSE else TRUE),
          distribution.model=dist_)
        m_i <- tryCatch(rugarch::ugarchfit(sp, data=as.numeric(a_df$value), solver="hybrid"),
                        error=function(e) NULL)
        list(model=m_i, forecast_fn=function(h){
          if(is.null(m_i)) return(rep(NA_real_,h))
          fc_i <- tryCatch(rugarch::ugarchforecast(m_i,n.ahead=h),error=function(e) NULL)
          if(is.null(fc_i)) return(rep(NA_real_,h))
          as.numeric(rugarch::fitted(fc_i))
        })
      }})
    cv_out <- .resample_cv_stat(df, horizon, cv_initial, cv_window, cv_skip, fit_fn, "garch")
    if(!is.null(cv_out)) cv_results <- cv_out
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic(); n_total <- nrow(df)
    n_ho <- max(horizon, floor(n_total*cv_holdout_frac))
    n_tr <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    m_ho <- tryCatch(rugarch::ugarchfit(spec, data=tr_df$value, solver="hybrid"),error=function(e) NULL)
    n_tst <- nrow(tst_df)
    fv    <- if(!is.null(m_ho)){
      fc_ho <- tryCatch(rugarch::ugarchforecast(m_ho, n.ahead=min(horizon,n_tst)),error=function(e) NULL)
      if(!is.null(fc_ho)) c(as.numeric(rugarch::fitted(fc_ho)), rep(NA_real_,max(0,n_tst-horizon)))
      else rep(NA_real_,n_tst)
    } else rep(NA_real_,n_tst)
    m_ho <- .safe_metrics(tst_df$value, fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f\n",m_ho$RMSE,m_ho$MAE))
    holdout_results <- tibble(fold=1L, h=seq_len(n_tst), date=tst_df$date,
      actual=tst_df$value, fitted=fv, residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec, eval_sec=eval_sec, total_sec=.toc(total_t0))
  cat(sprintf("\n✓ GARCH[%s] | total=%s\n",model_type,.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model_fit, spec=spec, model_type=model_type, p=p, q=q,
    distribution=distribution, forecast=forecast_df, cv_results=cv_results,
    holdout_results=holdout_results, data=df, horizon=horizon, runtime=runtime), class="auto_garch")
}


# ================================================================
# ██  PART 15 — UNIFIED OOS HELPERS (.get_oos / fitted_oos)
# ================================================================

.get_oos <- function(obj, strategy="recursive"){
  .rn <- function(df) if("fitted"%in%names(df)&&!"predicted"%in%names(df)) rename(df,predicted=fitted) else df
  if(inherits(obj,"auto_rnn")){
    if(!is.null(obj$cv_results)&&!is.null(obj$cv_results$oos_tbl)){
      oos <- obj$cv_results$oos_tbl
      if("strategy"%in%names(oos)&&strategy!="both") oos <- dplyr::filter(oos,strategy==!!strategy)
      return(oos)
    }
    if(!is.null(obj$holdout_results)){
      rows <- list()
      for(s in c("recursive","direct","nbeats")){
        tbl <- obj$holdout_results[[s]]
        if(!is.null(tbl)){tbl<-.rn(tbl);if(strategy=="both"||strategy==s) rows[[s]]<-tbl}
      }
      return(dplyr::bind_rows(rows))
    }
  }
  if(inherits(obj,c("auto_arima_ts","auto_ets_ts","auto_arfima_ts","auto_har_rv","auto_garch"))){
    if(!is.null(obj$cv_results)&&!is.null(obj$cv_results$oos_tbl)) return(.rn(obj$cv_results$oos_tbl))
    if(!is.null(obj$holdout_results)) return(.rn(obj$holdout_results))
  }
  stop("No OOS results found.")
}

fitted_oos <- function(object,...) UseMethod("fitted_oos")
for(.cls in c("auto_rnn","auto_arima_ts","auto_ets_ts","auto_arfima_ts","auto_har_rv","auto_garch"))
  assign(paste0("fitted_oos.",.cls), function(object,...) .get_oos(object,...))

predicted <- function(object,...) UseMethod("predicted")
.pred_default <- function(object, aggregate=FALSE,...){
  oos <- .get_oos(object)
  if(!aggregate) return(oos)
  has_s <- "strategy"%in%names(oos)
  oos %>% group_by(across(all_of(c("h",if(has_s)"strategy" else NULL)))) %>%
    summarise(mean_actual=mean(actual), mean_predicted=mean(predicted),
              RMSE=sqrt(mean(residual^2)), MAE=mean(abs(residual)), .groups="drop")
}
#' @export
predicted.auto_rnn <- function(object, strategy=c("both","recursive","direct"), aggregate=FALSE,...){
  oos <- .get_oos(object, match.arg(strategy))
  if(!aggregate) return(oos)
  has_s <- "strategy"%in%names(oos)
  oos %>% group_by(across(all_of(c("h",if(has_s)"strategy" else NULL)))) %>%
    summarise(mean_actual=mean(actual), mean_predicted=mean(predicted),
              RMSE=sqrt(mean(residual^2)), MAE=mean(abs(residual)), .groups="drop")
}
for(.cls in c("auto_arima_ts","auto_ets_ts","auto_arfima_ts","auto_har_rv","auto_garch"))
  assign(paste0("predicted.",.cls), .pred_default)


# ================================================================
# ██  PART 16 — REDESIGNED predict() — THREE CALL SIGNATURES
# ================================================================

#' @export
predict.auto_rnn <- function(object, horizon=NULL, new_data=NULL,
                              strategy=c("both","recursive","direct"),
                              level=95, ...){
  strategy <- match.arg(strategy); sc <- object$scaler; dev <- get_device()
  is_nb    <- object$model_type=="nbeats"
  freq_str <- infer_frequency(object$data$date)
  mdesc    <- sprintf("Auto%s", toupper(object$model_type))

  if(!is.null(new_data) && is.null(horizon)){
    # Signature 2: rolling OOS on test set
    stopifnot(is.data.frame(new_data), "date"%in%names(new_data))
    val_col   <- setdiff(names(new_data),"date")[1]
    test_vals <- new_data[[val_col]]; train_vals <- object$data$value
    res <- list()
    for(s in c("recursive","direct","nbeats")){
      if(s=="nbeats"&&!is_nb) next
      if(s!="nbeats"&&(!s%in%c(strategy,"both"))&&strategy!="both") next
      mdl <- if(s%in%c("recursive","nbeats")) object$model_recursive
             else if(!is.null(object$direct_models)) object$direct_models[[1]]
             else NULL
      if(is.null(mdl)) next
      tr_sc  <- scale_x(train_vals, sc); tst_sc <- scale_x(test_vals, sc)
      fv     <- .rolling_oos(mdl, tr_sc, tst_sc, object$lags, sc, dev)
      res_sd <- sd(test_vals-fv, na.rm=TRUE); ci_w <- qnorm(0.5+level/200)*res_sd
      res[[s]] <- .make_pred_tbl(new_data$date, fv, fv-ci_w, fv+ci_w,
                                  sprintf("%s [%s]",mdesc,s), "actual") %>% mutate(.strategy=s)
    }
    if(length(res)==0L) stop("No model available.")
    return(dplyr::bind_rows(res))
  }

  # Signature 3: horizon from tail of new_data
  if(!is.null(new_data) && !is.null(horizon)){
    h        <- as.integer(horizon)
    all_vals <- c(object$data$value, new_data[[setdiff(names(new_data),"date")[1]]])
    last_win <- tail(scale_x(all_vals, sc), object$lags)
    last_date <- max(new_data$date)
    fut_dates <- seq(last_date, by=freq_str, length.out=h+1L)[-1L]
    return(.predict_future(object, last_win, h, fut_dates, sc, dev, strategy, level, mdesc, is_nb))
  }

  # Signature 1: pure future forecast
  h        <- as.integer(horizon %||% object$horizon)
  last_win <- tail(scale_x(object$data$value, sc), object$lags)
  last_date <- max(object$data$date)
  fut_dates <- seq(last_date, by=freq_str, length.out=h+1L)[-1L]
  .predict_future(object, last_win, h, fut_dates, sc, dev, strategy, level, mdesc, is_nb)
}

.predict_future <- function(object, last_win, h, fut_dates, sc, dev, strategy, level, mdesc, is_nb){
  z_val   <- qnorm(0.5+level/200)
  res_sd  <- tryCatch({
    sv <- make_supervised(scale_x(object$data$value,sc), object$lags)
    pd <- predict_deep(object$model_recursive, sv$X, dev, is_nb)
    sd(unscale_x(sv$y,sc) - unscale_x(pd,sc))
  }, error=function(e) sd(object$data$value)*0.1)
  ci_w <- z_val*res_sd*sqrt(seq_len(h)); res <- list()
  if((strategy%in%c("recursive","both")||is_nb)&&!is.null(object$model_recursive)){
    fc <- if(is_nb) nbeats_forecast(object$model_recursive, last_win, h, sc, dev)
          else      recursive_forecast(object$model_recursive, last_win, h, sc, dev)
    strat_lbl <- if(is_nb)"nbeats" else "recursive"
    res[[strat_lbl]] <- .make_pred_tbl(fut_dates, fc, fc-ci_w, fc+ci_w,
      sprintf("%s [%s]",mdesc,strat_lbl), "prediction") %>% mutate(.strategy=strat_lbl)
  }
  if(strategy%in%c("direct","both")&&!is_nb&&!is.null(object$direct_models)&&
     length(object$direct_models)>=h){
    fc <- direct_forecast(object$direct_models[seq_len(h)], last_win, sc, dev)
    res$direct <- .make_pred_tbl(fut_dates, fc, fc-ci_w, fc+ci_w,
      sprintf("%s [direct]",mdesc), "prediction") %>% mutate(.strategy="direct")
  }
  if(length(res)==0L) stop("No models available.")
  dplyr::bind_rows(res)
}

#' @export
predict.auto_arima_ts <- function(object, horizon=NULL, new_data=NULL, level=NULL, xreg=NULL,...){
  h <- as.integer(horizon%||%object$horizon); lv <- level%||%object$level
  mdesc <- sprintf("Auto-ARIMA(%d,%d,%d)",object$model$arma[1],object$model$arma[6],object$model$arma[2])
  if(!is.null(new_data)&&is.null(horizon)){
    stopifnot(is.data.frame(new_data))
    all_df  <- dplyr::bind_rows(object$data%>%select(date,value),
                                 new_data%>%rename_with(~c("date","value"),everything()))
    all_ts  <- .df_to_ts(all_df); n_tr <- nrow(object$data); n_tst <- nrow(new_data)
    refit   <- tryCatch(forecast::Arima(all_ts,model=object$model),error=function(e) NULL)
    fv      <- if(!is.null(refit)){af<-as.numeric(fitted(refit));af[(n_tr+1L):(n_tr+n_tst)]}
               else tryCatch(as.numeric(forecast::forecast(object$model,h=n_tst)$mean),
                             error=function(e) rep(NA_real_,n_tst))
    res_sd  <- sd(fv-new_data[[2]],na.rm=TRUE)%||%sd(object$data$value)*0.1
    z_val   <- qnorm(0.5+(max(lv)/100+1)/2)
    return(.make_pred_tbl(new_data$date, fv, fv-z_val*res_sd, fv+z_val*res_sd, mdesc, "actual"))
  }
  base_model <- if(!is.null(new_data))
    tryCatch(forecast::Arima(.df_to_ts(new_data%>%rename_with(~c("date","value"),everything())),model=object$model),
             error=function(e) object$model)
  else object$model
  last_date <- if(!is.null(new_data)) max(new_data$date) else max(object$data$date)
  fut_dates <- seq(last_date,by=infer_frequency(object$data$date),length.out=h+1L)[-1L]
  fc <- forecast::forecast(base_model, h=h, level=lv, xreg=xreg)
  .make_pred_tbl(fut_dates, as.numeric(fc$mean),
                 as.numeric(fc$lower[,ncol(fc$lower)]),
                 as.numeric(fc$upper[,ncol(fc$upper)]), mdesc, "prediction")
}

#' @export
predict.auto_ets_ts <- function(object, horizon=NULL, new_data=NULL, level=NULL,...){
  h  <- as.integer(horizon%||%object$horizon); lv <- level%||%object$level
  mdesc <- sprintf("Auto-ETS(%s)", object$model$method)
  if(!is.null(new_data)&&is.null(horizon)){
    stopifnot(is.data.frame(new_data)); n_tst <- nrow(new_data)
    ts_nd <- .df_to_ts(new_data%>%rename_with(~c("date","value"),everything()))
    m_ho  <- tryCatch(forecast::ets(ts_nd, model=object$model$method),error=function(e) NULL)
    fv    <- if(!is.null(m_ho)) as.numeric(fitted(m_ho)) else rep(NA_real_,n_tst)
    res_sd <- sd(fv-new_data[[2]],na.rm=TRUE)%||%sd(object$data$value)*0.1
    z_val  <- qnorm(0.5+(max(lv)/100+1)/2)
    return(.make_pred_tbl(new_data$date, fv, fv-z_val*res_sd, fv+z_val*res_sd, mdesc, "actual"))
  }
  last_date <- if(!is.null(new_data)) max(new_data$date) else max(object$data$date)
  base_ts   <- if(!is.null(new_data)) .df_to_ts(new_data%>%rename_with(~c("date","value"),everything()))
               else object$ts_obj
  freq_str  <- infer_frequency(object$data$date)
  fut_dates <- seq(last_date, by=freq_str, length.out=h+1L)[-1L]
  m_fc      <- tryCatch(forecast::ets(base_ts,model=object$model$method),error=function(e) object$model)
  fc        <- forecast::forecast(m_fc, h=h, level=lv)
  .make_pred_tbl(fut_dates, as.numeric(fc$mean),
                 as.numeric(fc$lower[,ncol(fc$lower)]),
                 as.numeric(fc$upper[,ncol(fc$upper)]), mdesc, "prediction")
}

#' @export
predict.auto_arfima_ts <- function(object, horizon=NULL, new_data=NULL, level=NULL,...){
  h  <- as.integer(horizon%||%object$horizon); lv <- level%||%object$level
  mdesc <- sprintf("Auto-ARFIMA(d=%.4f)", object$d_hat)
  if(!is.null(new_data)&&is.null(horizon)){
    stopifnot(is.data.frame(new_data))
    all_df  <- dplyr::bind_rows(object$data%>%select(date,value),
                                 new_data%>%rename_with(~c("date","value"),everything()))
    all_ts  <- .df_to_ts(all_df); n_tr <- nrow(object$data); n_tst <- nrow(new_data)
    refit   <- tryCatch(forecast::Arima(all_ts,model=object$model),error=function(e) NULL)
    fv      <- if(!is.null(refit)){af<-as.numeric(fitted(refit));af[(n_tr+1L):(n_tr+n_tst)]}
               else rep(NA_real_,n_tst)
    res_sd  <- sd(fv-new_data[[2]],na.rm=TRUE)%||%sd(object$data$value)*0.1
    z_val   <- qnorm(0.5+(max(lv)/100+1)/2)
    return(.make_pred_tbl(new_data$date, fv, fv-z_val*res_sd, fv+z_val*res_sd, mdesc, "actual"))
  }
  last_date <- if(!is.null(new_data)) max(new_data$date) else max(object$data$date)
  freq_str  <- infer_frequency(object$data$date)
  fut_dates <- seq(last_date, by=freq_str, length.out=h+1L)[-1L]
  fc <- tryCatch(forecast::forecast(object$model,h=h,level=lv),error=function(e) NULL)
  if(is.null(fc)) return(.make_pred_tbl(fut_dates,rep(NA_real_,h),NA_real_,NA_real_,mdesc,"prediction"))
  .make_pred_tbl(fut_dates, as.numeric(fc$mean),
                 as.numeric(fc$lower[,ncol(fc$lower)]),
                 as.numeric(fc$upper[,ncol(fc$upper)]), mdesc, "prediction")
}

#' @export
predict.auto_har_rv <- function(object, horizon=NULL, new_data=NULL,...){
  h <- as.integer(horizon%||%object$horizon)
  mdesc <- sprintf("HAR-RV[%s]",object$variant)
  .har_fc_pred <- function(rv_h, h_steps){
    preds <- numeric(h_steps)
    for(step in seq_len(h_steps)){
      n_h <- length(rv_h); lag1 <- rv_h[n_h]
      lagw <- mean(rv_h[max(1,n_h-object$w_lags+1):n_h])
      lagm <- mean(rv_h[max(1,n_h-object$m_lags+1):n_h])
      nd   <- data.frame(RV_d=lag1, RV_w=lagw, RV_m=lagm)
      for(nm in setdiff(names(coef(object$model)),c("(Intercept)","RV_d","RV_w","RV_m")))
        nd[[nm]] <- 0
      p <- tryCatch(predict(object$model,newdata=nd),error=function(e) NA_real_)
      preds[step] <- if(object$log_transform) exp(p) else p
      rv_h <- c(rv_h, if(object$log_transform) p else preds[step])
    }; preds
  }
  if(!is.null(new_data)&&is.null(horizon)){
    rv_hist <- if(object$log_transform) log(pmax(object$data$value,1e-10)) else object$data$value
    rv_test <- if(object$log_transform) log(pmax(new_data[[2]],1e-10)) else new_data[[2]]
    n_tst <- length(rv_test); fv <- numeric(n_tst)
    for(i in seq_len(n_tst)){
      rv_cur <- c(rv_hist, if(i>1) rv_test[seq_len(i-1)] else NULL)
      fv[i]  <- .har_fc_pred(tail(rv_cur,max(object$m_lags,60L)), 1L)[1L]
    }
    res_sd <- sd(fv-new_data[[2]],na.rm=TRUE)%||%sd(object$data$value)*0.1
    return(.make_pred_tbl(new_data$date, fv, fv-1.96*res_sd, fv+1.96*res_sd, mdesc, "actual"))
  }
  base_date <- if(!is.null(new_data)) max(new_data$date) else max(object$data$date)
  freq_str  <- infer_frequency(object$data$date)
  fut_dates <- seq(base_date, by=freq_str, length.out=h+1L)[-1L]
  rv_h <- if(object$log_transform) log(pmax(tail(object$data$value,max(object$m_lags,60L)),1e-10))
          else tail(object$data$value, max(object$m_lags,60L))
  fc_vals <- .har_fc_pred(rv_h, h)
  .make_pred_tbl(fut_dates, fc_vals, NA_real_, NA_real_, mdesc, "prediction")
}

#' @export
predict.auto_garch <- function(object, horizon=NULL, new_data=NULL,...){
  h <- as.integer(horizon%||%object$horizon)
  mdesc <- sprintf("GARCH[%s(%d,%d)]",object$model_type,object$p,object$q)
  if(!is.null(new_data)&&is.null(horizon)){
    stopifnot(is.data.frame(new_data))
    all_vals <- c(object$data$value, new_data[[setdiff(names(new_data),"date")[1]]])
    n_tr <- nrow(object$data); n_tst <- nrow(new_data)
    refit <- tryCatch(rugarch::ugarchfit(object$spec,data=all_vals,solver="hybrid"),error=function(e) NULL)
    if(!is.null(refit)){
      fv  <- as.numeric(rugarch::fitted(refit))[(n_tr+1L):(n_tr+n_tst)]
      sig <- as.numeric(rugarch::sigma(refit))[(n_tr+1L):(n_tr+n_tst)]
    } else { fv <- rep(NA_real_,n_tst); sig <- rep(NA_real_,n_tst) }
    return(.make_pred_tbl(new_data$date, fv, fv-1.96*sig, fv+1.96*sig, mdesc, "actual"))
  }
  base_model <- if(!is.null(new_data)){
    all_vals <- c(object$data$value, new_data[[setdiff(names(new_data),"date")[1]]])
    tryCatch(rugarch::ugarchfit(object$spec,data=all_vals,solver="hybrid"),error=function(e) object$model)
  } else object$model
  last_date <- if(!is.null(new_data)) max(new_data$date) else max(object$data$date)
  freq_str  <- infer_frequency(object$data$date)
  fut_dates <- seq(last_date, by=freq_str, length.out=h+1L)[-1L]
  fc  <- rugarch::ugarchforecast(base_model, n.ahead=h)
  mfc <- as.numeric(rugarch::fitted(fc)); sfc <- as.numeric(rugarch::sigma(fc))
  .make_pred_tbl(fut_dates, mfc, mfc-1.96*sfc, mfc+1.96*sfc, mdesc, "prediction")
}


# ================================================================
# ██  PART 17 — modeltime INTEGRATION + tf_calibrate / tf_accuracy
# ================================================================

#' @export
tf_modeltime_table <- function(...){
  results <- list(...)
  nms <- if(!is.null(names(results))&&any(names(results)!="")) names(results)
         else vapply(results, function(r){
    if(inherits(r,"auto_rnn"))        sprintf("Auto%s",toupper(r$model_type))
    else if(inherits(r,"auto_arima_ts"))  "AutoARIMA"
    else if(inherits(r,"auto_ets_ts"))    "AutoETS"
    else if(inherits(r,"auto_arfima_ts")) "AutoARFIMA"
    else if(inherits(r,"auto_har_rv"))    sprintf("HAR-%s",r$variant)
    else if(inherits(r,"auto_garch"))     sprintf("%s(%d,%d)",r$model_type,r$p,r$q)
    else "Unknown"
  }, character(1L))
  model_list <- lapply(seq_along(results), function(i){
    r <- results[[i]]; nm <- nms[i]
    wf <- workflows::workflow()
    wf <- workflows::add_model(wf, parsnip::null_model()%>%parsnip::set_engine("parsnip"))
    wf$.__tf_result <- r; wf$.__tf_label <- nm
    class(wf) <- c("tf_workflow", class(wf)); wf
  })
  modeltime::modeltime_table(!!!model_list)
}

#' @export
predict.tf_workflow <- function(object, new_data, type="numeric",...){
  r   <- object$.__tf_result
  tbl <- predict(r, new_data=new_data,...)
  tibble(.pred=tbl$.value)
}

#' @export
tf_calibrate <- function(results, new_data){
  if(!is.list(results)||inherits(results,c("auto_rnn","auto_arima_ts","auto_ets_ts",
                                            "auto_arfima_ts","auto_har_rv","auto_garch")))
    results <- list(results)
  stopifnot(is.data.frame(new_data), ncol(new_data)>=2)
  val_col <- setdiff(names(new_data),"date")[1]; actuals <- new_data[[val_col]]; dates <- new_data$date
  rows <- lapply(seq_along(results), function(i){
    r  <- results[[i]]; nm <- names(results)[i]%||%sprintf("Model_%d",i)
    tbl <- tryCatch(predict(r, new_data=new_data),
                    error=function(e){warning(sprintf("tf_calibrate [%s]: %s",nm,e$message));NULL})
    if(is.null(tbl)) return(NULL)
    tbl %>% mutate(.model_id=i, .model_name=nm, .actual=actuals, .residuals=actuals-.value)
  })
  cal <- dplyr::bind_rows(Filter(Negate(is.null),rows))
  attr(cal,"new_data") <- new_data; attr(cal,"results") <- results
  class(cal) <- c("tf_calibrated",class(cal)); cal
}

#' @export
print.tf_calibrated <- function(x,...){
  cat("\n── tf_calibrated ──────────────────────────────────────────\n")
  cat(sprintf("  Models: %d | Test rows: %d\n",
              dplyr::n_distinct(x$.model_id), nrow(attr(x,"new_data"))))
  print(tf_accuracy(x)); invisible(x)
}

#' @export
tf_accuracy <- function(calibrated){
  if(!inherits(calibrated,"tf_calibrated")) stop("Need tf_calibrated output.")
  has_s <- ".strategy"%in%names(calibrated)
  calibrated %>%
    group_by(.model_id, .model_name, .model_desc,
             if(has_s) .strategy else NULL) %>%
    summarise(n=dplyr::n(), MAE=mean(abs(.residuals),na.rm=TRUE),
              RMSE=sqrt(mean(.residuals^2,na.rm=TRUE)),
              MAPE=mean(abs(.residuals/.actual)*100,na.rm=TRUE),
              SMAPE=mean(200*abs(.residuals)/(abs(.actual)+abs(.value)),na.rm=TRUE),
              RSQ=tryCatch(cor(.actual,.value,use="complete.obs")^2,error=function(e) NA_real_),
              MBE=mean(.residuals,na.rm=TRUE), .groups="drop") %>%
    arrange(RMSE)
}

#' @export
tf_forecast <- function(calibrated, horizon=NULL, new_data=NULL){
  if(!inherits(calibrated,"tf_calibrated")) stop("Need tf_calibrated.")
  results <- attr(calibrated,"results")
  rows <- lapply(seq_along(results), function(i){
    r  <- results[[i]]; nm <- names(results)[i]%||%sprintf("Model_%d",i)
    tryCatch(predict(r, horizon=horizon, new_data=new_data) %>%
               mutate(.model_id=i, .model_name=nm),
             error=function(e){warning(sprintf("tf_forecast [%s]: %s",nm,e$message));NULL})
  })
  dplyr::bind_rows(Filter(Negate(is.null),rows)) %>%
    select(.model_id,.model_name,.model_desc,.key,date,.value,.conf_lo,.conf_hi,everything())
}

#' @export
tf_forecast_combined <- function(calibrated, horizon=NULL, new_data=NULL){
  actual_tbl <- attr(calibrated,"new_data") %>%
    mutate(.key="actual", .value=.data[[setdiff(names(.),"date")[1]]],
           .conf_lo=NA_real_, .conf_hi=NA_real_, .model_desc="Actual",
           .model_id=0L, .model_name="Actual") %>%
    select(.model_id,.model_name,.model_desc,.key,date,.value,.conf_lo,.conf_hi)
  dplyr::bind_rows(actual_tbl, tf_forecast(calibrated, horizon=horizon, new_data=new_data))
}

cv_performance <- function(object,...) UseMethod("cv_performance")
#' @export
cv_performance.auto_rnn <- function(object, strategy=c("both","recursive","direct"),
                                     by_horizon=TRUE,...){
  strategy <- match.arg(strategy); oos <- .get_oos(object, strategy)
  has_s <- "strategy"%in%names(oos)
  pc    <- if("fitted"%in%names(oos))"fitted" else "predicted"
  per_h <- if(has_s) oos%>%group_by(h,strategy) else oos%>%group_by(h)
  per_h <- per_h%>%summarise(RMSE=sqrt(mean(residual^2,na.rm=TRUE)),
    MAE=mean(abs(residual),na.rm=TRUE), MAPE=mean(abs(residual/actual)*100,na.rm=TRUE),
    SMAPE=mean(200*abs(residual)/(abs(actual)+abs(.data[[pc]])),na.rm=TRUE),
    MBE=mean(residual,na.rm=TRUE), .groups="drop")
  cat("\n── OOS CV (modeltime.resample) ──\n"); print(per_h, n=Inf)
  if(!is.null(object$cv_results)&&!is.null(object$cv_results$cv_metrics)){
    cat("\n── Overall ──\n"); print(object$cv_results$cv_metrics)}
  invisible(list(by_horizon=per_h, overall=object$cv_results$cv_metrics, raw_oos=oos))
}
for(.cls in c("auto_arima_ts","auto_ets_ts","auto_arfima_ts","auto_har_rv","auto_garch"))
  assign(paste0("cv_performance.",.cls), function(object,...){
    cat(sprintf("\n── %s resample CV ──\n",class(object)[1]))
    if(!is.null(object$cv_results)) print(object$cv_results$cv_metrics)
    invisible(object$cv_results)
  })

# ================================================================
# ██  PART 18 — S3: print / summary / coef / fitted / residuals
# ================================================================

#' @export
print.auto_rnn <- function(x, digits=4L, n_fc=5L,...){
  mt <- toupper(x$model_type); rt <- x$runtime
  cat("\n",.box(),"\n  Auto",mt,"  (torchforecast v6)\n",.box(),"\n",sep="")
  cat(.hdr("Data"),"\n")
  cat(sprintf("  Obs: %d | %s→%s | freq=%s\n", nrow(x$data),
              format(min(x$data$date)), format(max(x$data$date)), infer_frequency(x$data$date)))
  cat(.hdr("Model"),"\n")
  cat(sprintf("  arch=%s | strategy=%s | h=%d | lags=%d\n",mt,x$strategy,x$horizon,x$lags))
  hp <- x$best_params
  if(!is.null(hp)) cat(sprintf("  HPs: hidden=%d|layers=%d|dropout=%.3f|dense=%d|lr=%.2e|batch=%d\n",
    hp$hidden_size,hp$num_layers,hp$dropout,hp$dense_units,hp$lr,hp$batch_size))
  cat(.hdr("predict() signatures"),"\n")
  cat(sprintf("  predict(model, horizon=%d)         # future forecast\n",x$horizon))
  cat("  predict(model, new_data=test_df)   # rolling OOS on test\n")
  cat("  predict(model, new_data, horizon)  # forecast from test tail\n")
  cat(.hdr("CV engine: modeltime.resample"),"\n")
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$cv_metrics))
    print(x$cv_results$cv_metrics, n=Inf)
  else if(!is.null(x$holdout_results))
    cat(sprintf("  Hold-out | train=%d | test=%d\n",
                x$holdout_results$n_train%||%"?",x$holdout_results$n_test%||%"?"))
  cat(.hdr("BO engine: ParBayesianOptimization"),"\n")
  cat(.hdr("Runtime"),"\n")
  cat(sprintf("  BO=%s | Eval=%s | Final=%s | TOTAL=%s\n",
    .fmt_elapsed(rt$bo_sec%||%0), .fmt_elapsed((rt$cv_sec%||%rt$holdout_sec)%||%0),
    .fmt_elapsed(rt$final_train_sec%||%0), .fmt_elapsed(rt$total_sec%||%0)))
  cat(.box(),"\n\n"); invisible(x)
}

.print_stat <- function(x, label, extra=""){
  rt <- x$runtime
  cat("\n",.box(),sprintf("\n  %s  (torchforecast v6)\n",label),.box(),"\n",sep="")
  cat(sprintf("  CV: modeltime.resample | BO: ParBayesianOptimization\n"))
  if(nchar(extra)>0) cat(sprintf("  %s\n",extra))
  cat("  predict() signatures: horizon | new_data | (new_data+horizon)\n")
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$cv_metrics)){
    cat("  Resample CV overall:\n"); print(x$cv_results$cv_metrics)
  }
  cat(sprintf("  total=%s\n",.fmt_elapsed(rt$total_sec)))
  cat(.box(),"\n\n"); invisible(x)
}

#' @export
print.auto_arima_ts <- function(x,...){
  m <- x$model
  .print_stat(x,"Auto-ARIMA",
    sprintf("ARIMA(%d,%d,%d) AICc=%.2f",m$arma[1],m$arma[6],m$arma[2],m$aicc))
}
#' @export
print.auto_ets_ts <- function(x,...){
  .print_stat(x,"Auto-ETS",sprintf("ETS(%s) AICc=%.2f",x$model$method,x$model$aicc))
}
#' @export
print.auto_arfima_ts <- function(x,...) .print_stat(x,"Auto-ARFIMA",sprintf("d=%.6f",x$d_hat))
#' @export
print.auto_har_rv    <- function(x,...) .print_stat(x,sprintf("HAR-RV [%s]",x$variant))
#' @export
print.auto_garch     <- function(x,...){
  ic <- rugarch::infocriteria(x$model)
  .print_stat(x,sprintf("GARCH [%s(%d,%d)]",x$model_type,x$p,x$q),
              sprintf("AIC=%.4f dist=%s",ic[1],x$distribution))
}

#' @export
summary.auto_rnn        <- function(object,...){ print(object,...); invisible(object) }
#' @export
summary.auto_arima_ts   <- function(object,...){ print(object,...); print(summary(object$model)); invisible(object) }
#' @export
summary.auto_ets_ts     <- function(object,...){ print(object,...); print(summary(object$model)); invisible(object) }
#' @export
summary.auto_arfima_ts  <- function(object,...){ print(object,...)
  tryCatch(print(summary(object$model)),error=function(e) print(summary(object$fd_fit)))
  invisible(object) }
#' @export
summary.auto_har_rv     <- function(object,...){ print(object,...); print(summary(object$model)); invisible(object) }
#' @export
summary.auto_garch      <- function(object,...){ print(object,...); rugarch::show(object$model); invisible(object) }

#' @export
coef.auto_rnn <- function(object,...) unlist(lapply(object$best_params, as.numeric))

#' @export
fitted.auto_rnn <- function(object,...){
  if(is.null(object$model_recursive)) stop("No recursive model.")
  is_nb <- object$model_type=="nbeats"
  sv    <- make_supervised(scale_x(object$data$value, object$scaler), object$lags)
  pred  <- predict_deep(object$model_recursive, sv$X, get_device(), is_nb)
  tibble(date=object$data$date[(object$lags+1L):nrow(object$data)],
         fitted=unscale_x(pred,object$scaler),
         actual=object$data$value[(object$lags+1L):nrow(object$data)]) %>%
    mutate(residual=actual-fitted)
}
#' @export
fitted.auto_arima_ts <- function(object,...){
  fv <- as.numeric(fitted(object$model)); n <- min(length(fv),nrow(object$data))
  tibble(date=object$data$date[seq_len(n)], fitted=fv[seq_len(n)],
         actual=object$data$value[seq_len(n)]) %>% mutate(residual=actual-fitted)
}
#' @export
fitted.auto_ets_ts <- function(object,...){
  fv <- as.numeric(fitted(object$model)); n <- min(length(fv),nrow(object$data))
  tibble(date=object$data$date[seq_len(n)], fitted=fv[seq_len(n)],
         actual=object$data$value[seq_len(n)]) %>% mutate(residual=actual-fitted)
}
#' @export
fitted.auto_arfima_ts <- function(object,...){
  fv <- tryCatch(as.numeric(fitted(object$model)), error=function(e) rep(NA_real_,nrow(object$data)))
  n  <- min(length(fv),nrow(object$data))
  tibble(date=object$data$date[seq_len(n)], fitted=fv[seq_len(n)],
         actual=object$data$value[seq_len(n)]) %>% mutate(residual=actual-fitted)
}
#' @export
fitted.auto_har_rv <- function(object,...){
  fv <- as.numeric(fitted(object$model))
  tibble(date=object$data$date[seq_along(fv)],
         fitted=if(object$log_transform) exp(fv) else fv,
         actual=object$data$value[seq_along(fv)]) %>% mutate(residual=actual-fitted)
}
#' @export
fitted.auto_garch <- function(object,...){
  fv <- as.numeric(rugarch::fitted(object$model)); sv <- as.numeric(rugarch::sigma(object$model))
  tibble(date=object$data$date[seq_along(fv)], fitted=fv, sigma=sv,
         actual=object$data$value[seq_along(fv)]) %>% mutate(residual=actual-fitted)
}

#' @export
residuals.auto_rnn        <- function(object,...) fitted(object)$residual
#' @export
residuals.auto_arima_ts   <- function(object,...) as.numeric(residuals(object$model))
#' @export
residuals.auto_ets_ts     <- function(object,...) as.numeric(residuals(object$model))
#' @export
residuals.auto_arfima_ts  <- function(object,...) tryCatch(as.numeric(residuals(object$model)),
                               error=function(e) as.numeric(residuals(object$fd_fit)))
#' @export
residuals.auto_har_rv     <- function(object,...) as.numeric(residuals(object$model))
#' @export
residuals.auto_garch      <- function(object, standardized=TRUE,...){
  if(standardized) as.numeric(rugarch::residuals(object$model,standardize=TRUE))
  else             as.numeric(rugarch::residuals(object$model))
}

#' @export
format.auto_rnn <- function(x,...) sprintf("<auto_%s|n=%d|h=%d|%s|t=%s>",
  x$model_type,nrow(x$data),x$horizon,
  if(!is.null(x$cv_results))"resample-CV" else "holdout",
  .fmt_elapsed(x$runtime$total_sec%||%NA))
#' @export
autoplot.auto_rnn       <- function(object,...){ p<-plot(object,...); invisible(p) }
#' @export
autoplot.auto_arima_ts  <- function(object,...){ p<-plot(object,...); invisible(p) }
#' @export
autoplot.auto_ets_ts    <- function(object,...){ p<-plot(object,...); invisible(p) }
#' @export
autoplot.auto_arfima_ts <- function(object,...){ p<-plot(object,...); invisible(p) }


# ================================================================
# ██  PART 19 — PLOT METHODS
# ================================================================

.fc_panel <- function(hist_df, fc_all, label, colour, bt){
  p <- ggplot()+
    geom_line(data=hist_df, aes(date,value), colour=.PAL$actual, linewidth=0.8)+
    geom_vline(xintercept=as.numeric(max(hist_df$date)), linetype="dashed", colour=.PAL$neutral)
  if(!is.null(fc_all)&&nrow(fc_all)>0){
    sp <- c(recursive=.PAL$recursive,direct=.PAL$direct,nbeats=.PAL$nbeats,model=colour)
    has_s <- "strategy"%in%names(fc_all)
    p <- p+
      geom_ribbon(data=fc_all, aes(date,ymin=lower_95,ymax=upper_95,
                                    fill=if(has_s) strategy else "model"),alpha=0.15)+
      geom_line(data=fc_all, aes(date,forecast, colour=if(has_s) strategy else "model"),
                linewidth=1, linetype="dashed")+
      geom_point(data=fc_all, aes(date,forecast, colour=if(has_s) strategy else "model"),
                 size=2, shape=21, fill="white", stroke=1.1)+
      scale_colour_manual(values=sp, name="Strategy")+scale_fill_manual(values=sp)
  }
  p+labs(title=sprintf("%s Forecast",label),
         subtitle="Dashed=forecast | Band=95%CI | CV: modeltime.resample",
         x=NULL, y="Value")+bt()
}

#' @export
plot.auto_rnn <- function(x, which=NULL, last_n=NULL,...){
  mt      <- toupper(x$model_type); panels <- list()
  hist_df <- if(!is.null(last_n)) tail(x$data,last_n) else x$data
  fc_all  <- dplyr::bind_rows(Filter(Negate(is.null),list(x$forecast_recursive,x$forecast_direct)))
  panels$forecast <- .fc_panel(hist_df, fc_all, sprintf("Auto%s",mt), .PAL[[mt]], .bt)
  if(!is.null(x$history_recursive)&&nrow(x$history_recursive)>0){
    h <- x$history_recursive
    best_ep <- h$epoch[which.min(replace(h$val_loss,is.na(h$val_loss),Inf))]
    panels$training <- h%>%select(epoch,Training=train_loss,Validation=val_loss)%>%
      pivot_longer(-epoch,names_to="split",values_to="MSE")%>%filter(!is.na(MSE))%>%
      ggplot(aes(epoch,MSE,colour=split))+
      geom_vline(xintercept=best_ep,linetype="dotted",colour="grey60")+geom_line(linewidth=0.85)+
      scale_colour_manual(values=c(Training=.PAL$train,Validation=.PAL$val),name=NULL)+
      labs(title="Training Loss [ParBayesOpt best model]",x="Epoch",y="MSE")+.bt()
  }
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$oos_tbl)){
    oos     <- x$cv_results$oos_tbl
    has_s   <- "strategy"%in%names(oos)
    by_fold <- oos%>%group_by(across(all_of(c("fold",if(has_s)"strategy" else NULL))))%>%
      summarise(RMSE=sqrt(mean(residual^2,na.rm=TRUE)),.groups="drop")
    panels$cv_fold <- ggplot(by_fold,aes(fold,RMSE,colour=if(has_s) strategy else "model",
                                          group=if(has_s) strategy else "model"))+
      geom_line(linewidth=0.8)+geom_point(size=2,shape=21,fill="white",stroke=1)+
      scale_colour_manual(values=c(recursive=.PAL$recursive,direct=.PAL$direct,
                                   nbeats=.PAL$nbeats,model=.PAL[[mt]]%||%.PAL$recursive),name=NULL)+
      labs(title="Resample RMSE per Split (modeltime.resample)",x="Split",y="RMSE")+.bt()
  }
  fv <- tryCatch(fitted(x),error=function(e) NULL)
  if(!is.null(fv))
    panels$residuals <- ggplot(fv,aes(date,residual))+
      geom_hline(yintercept=0,colour="grey40")+
      geom_hline(yintercept=c(-2,2)*sd(fv$residual,na.rm=TRUE),linetype="dashed",
                 colour=.PAL$direct,alpha=0.7)+
      geom_point(colour=.PAL[[mt]]%||%.PAL$recursive,alpha=0.5,size=1.2)+
      geom_smooth(method="loess",formula=y~x,se=FALSE,colour=.PAL$direct,linewidth=0.7)+
      labs(title="In-Sample Residuals",x=NULL,y="Residual")+.bt()
  sel <- if(is.null(which)) seq_along(panels) else which
  pw  <- patchwork::wrap_plots(panels[sel],ncol=2L)+
    patchwork::plot_annotation(title=sprintf("Auto%s Dashboard (v6 | modeltime.resample)",mt),
      theme=theme(plot.title=element_text(face="bold",size=14,hjust=0.5)))
  print(pw); invisible(pw)
}

.stat_plot <- function(x, label, colour){
  panels <- list(); hist_df <- x$data; fc <- x$forecast
  fc_all <- fc%>%mutate(strategy="model")
  if(!"lower_95"%in%names(fc_all)) fc_all$lower_95 <- NA_real_
  if(!"upper_95"%in%names(fc_all)) fc_all$upper_95 <- NA_real_
  panels$forecast <- .fc_panel(hist_df, fc_all, label, colour, .bt)
  fv <- tryCatch(fitted(x),error=function(e) NULL)
  if(!is.null(fv)&&"residual"%in%names(fv))
    panels$residuals <- filter(fv,!is.na(residual))%>%ggplot(aes(date,residual))+
      geom_hline(yintercept=0,colour="grey40")+geom_line(colour=colour,linewidth=0.7,alpha=0.7)+
      labs(title="In-Sample Residuals",x=NULL,y="Residual")+.bt()
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$oos_tbl)){
    oos <- x$cv_results$oos_tbl
    by_fold <- oos%>%group_by(fold)%>%
      summarise(RMSE=sqrt(mean(residual^2,na.rm=TRUE)),MAE=mean(abs(residual),na.rm=TRUE),.groups="drop")%>%
      pivot_longer(c(RMSE,MAE),names_to="metric",values_to="value")
    panels$by_fold <- ggplot(by_fold,aes(fold,value,colour=metric))+geom_line(linewidth=0.9)+
      geom_point(size=2,shape=21,fill="white",stroke=1.2)+
      scale_colour_manual(values=c(RMSE=colour,MAE=.PAL$direct),name=NULL)+
      labs(title="Resample Error per Split (modeltime.resample)",x="Split",y="Error")+.bt()
  }
  pw <- patchwork::wrap_plots(panels,ncol=2L)+
    patchwork::plot_annotation(title=sprintf("%s Dashboard (v6 | modeltime.resample)",label),
      theme=theme(plot.title=element_text(face="bold",size=14,hjust=0.5)))
  print(pw); invisible(pw)
}
#' @export
plot.auto_arima_ts  <- function(x,...) .stat_plot(x,sprintf("Auto-ARIMA(%d,%d,%d)",x$model$arma[1],x$model$arma[6],x$model$arma[2]),.PAL$arima)
#' @export
plot.auto_ets_ts    <- function(x,...) .stat_plot(x,sprintf("Auto-ETS(%s)",x$model$method),.PAL$ets)
#' @export
plot.auto_arfima_ts <- function(x,...) .stat_plot(x,sprintf("Auto-ARFIMA(d=%.4f)",x$d_hat),.PAL$arfima)
#' @export
plot.auto_har_rv    <- function(x,...) .stat_plot(x,sprintf("HAR-RV[%s]",x$variant),.PAL$har)
#' @export
plot.auto_garch     <- function(x,...) .stat_plot(x,sprintf("GARCH[%s(%d,%d)]",x$model_type,x$p,x$q),.PAL$garch)


# ================================================================
# ██  PART 20 — plot_oos_fit()
# ================================================================

plot_oos_fit <- function(object,...) UseMethod("plot_oos_fit")

.oos_dash <- function(oos, hist_df, title, eval_label, model_label, ncol=2L){
  bt <- .bt()
  if("fitted"%in%names(oos)&&!"predicted"%in%names(oos)) oos <- rename(oos,predicted=fitted)
  if(!"residual"%in%names(oos)) oos <- mutate(oos,residual=actual-predicted)
  has_s    <- "strategy"%in%names(oos)&&dplyr::n_distinct(oos$strategy)>1L
  has_h    <- "h"%in%names(oos)
  has_date <- "date"%in%names(oos)&&!all(is.na(oos$date))
  n_origins <- if("fold"%in%names(oos)) dplyr::n_distinct(oos$fold) else 1L
  s_pal    <- c(recursive=.PAL$recursive,direct=.PAL$direct,nbeats=.PAL$nbeats,
                model=.PAL$arima,arima=.PAL$arima,arfima=.PAL$arfima,
                ets=.PAL$ets,har=.PAL$har,garch=.PAL$garch)
  panels   <- list()
  if(has_date){
    grp_v <- c("date",if(has_s)"strategy" else NULL)
    agg   <- oos%>%group_by(across(all_of(grp_v)))%>%
      summarise(mean_actual=mean(actual,na.rm=TRUE),mean_predicted=mean(predicted,na.rm=TRUE),
                sd_predicted=sd(predicted,na.rm=TRUE),.groups="drop")
    p1 <- ggplot()
    if(!is.null(hist_df)&&nrow(hist_df)>0)
      p1 <- p1+geom_line(data=hist_df,aes(date,value),colour="grey40",linewidth=0.65,alpha=0.55)+
        geom_vline(xintercept=as.numeric(min(oos$date,na.rm=TRUE)),linetype="dashed",colour="grey60")
    if(has_s) p1 <- p1+geom_ribbon(data=agg,aes(date,ymin=mean_predicted-sd_predicted,
                                                   ymax=mean_predicted+sd_predicted,fill=strategy),alpha=0.12)
    p1 <- p1+
      geom_line(data=agg,aes(date,mean_actual),colour="grey20",linewidth=1.0)+
      geom_line(data=agg,aes(date,mean_predicted,colour=if(has_s) strategy else "model"),
                linewidth=1.0,linetype="dashed")+
      geom_point(data=agg,aes(date,mean_predicted,colour=if(has_s) strategy else "model"),
                 size=1.8,shape=21,fill="white",stroke=1)+
      scale_colour_manual(values=s_pal,name=NULL)+scale_fill_manual(values=s_pal)+
      labs(title="Actual vs Fitted (OOS)",
           subtitle=sprintf("%s | black=actual, dashed=fitted | %d splits",eval_label,n_origins),
           x=NULL,y="Value")+bt
    panels$time_series <- p1
  }
  panels$scatter <- ggplot(oos,aes(actual,predicted,colour=if(has_s) strategy else "model"))+
    geom_abline(slope=1,intercept=0,colour="grey40",linetype="dashed")+
    geom_point(alpha=0.5,size=1.5)+scale_colour_manual(values=s_pal,name=NULL)+
    labs(title="Actual vs Fitted Scatter",x="Actual",y="Fitted")+coord_equal()+bt
  if(has_date){
    res_agg <- oos%>%group_by(across(all_of(c("date",if(has_s)"strategy" else NULL))))%>%
      summarise(mean_res=mean(residual,na.rm=TRUE),.groups="drop")
    r_sd <- sd(oos$residual,na.rm=TRUE)
    panels$resid_time <- ggplot(res_agg,aes(date,mean_res,colour=if(has_s) strategy else "model"))+
      geom_hline(yintercept=0,colour="grey40")+
      geom_hline(yintercept=c(-2,2)*r_sd,linetype="dashed",colour=.PAL$direct,alpha=0.7)+
      geom_line(linewidth=0.6,alpha=0.8)+geom_point(alpha=0.5,size=1.2)+
      geom_smooth(method="loess",formula=y~x,se=FALSE,linewidth=0.75,colour=.PAL$direct)+
      scale_colour_manual(values=s_pal,name=NULL)+
      labs(title="OOS Residuals over Time",subtitle="±2σ | LOESS",x=NULL,y="Residual")+bt
  }
  panels$error_dist <- ggplot(oos,aes(residual,fill=if(has_s) strategy else "model"))+
    geom_histogram(bins=30L,alpha=0.65,position="identity",colour="white")+
    geom_vline(xintercept=0,colour="grey30",linetype="dashed")+
    scale_fill_manual(values=s_pal,name=NULL)+
    labs(title="OOS Error Distribution",x="Residual",y="Count")+bt
  if("fold"%in%names(oos)&&dplyr::n_distinct(oos$fold)>5L){
    fold_r <- oos%>%group_by(across(all_of(c("fold",if(has_s)"strategy" else NULL))))%>%
      summarise(RMSE=sqrt(mean(residual^2,na.rm=TRUE)),.groups="drop")
    panels$fold_box <- ggplot(fold_r,aes(x=if(has_s) strategy else "model",y=RMSE,
                                          fill=if(has_s) strategy else "model"))+
      geom_boxplot(alpha=0.65,show.legend=FALSE)+
      geom_jitter(width=0.15,alpha=0.5,size=1.5,show.legend=FALSE)+
      scale_fill_manual(values=s_pal)+
      labs(title=sprintf("RMSE across Splits (%d | modeltime.resample)",dplyr::n_distinct(oos$fold)),
           x=NULL,y="RMSE")+bt
  }
  pw <- patchwork::wrap_plots(panels,ncol=ncol)+
    patchwork::plot_annotation(title=title,
      caption=sprintf("Engine: %s  ·  %s  ·  torchforecast v6",eval_label,model_label),
      theme=theme(plot.title=element_text(face="bold",size=14,hjust=0.5),
                  plot.caption=element_text(size=9,colour="grey50",hjust=0.5)))
  print(pw); invisible(pw)
}

#' @export
plot_oos_fit.auto_rnn <- function(object,strategy=c("both","recursive","direct"),last_n=NULL,ncol=2L,title=NULL,...){
  strategy<-match.arg(strategy); oos<-.get_oos(object,strategy)
  if(nrow(oos)==0L) stop("Empty OOS tibble.")
  hist_df<-if(!is.null(last_n)) tail(object$data,last_n) else object$data
  if(is.null(title)) title<-sprintf("Auto%s — OOS (v6)",toupper(object$model_type))
  ev<-if(!is.null(object$cv_results))"modeltime.resample" else "Hold-out"
  .oos_dash(oos,hist_df,title,ev,toupper(object$model_type),ncol)
}
.make_oos_plot <- function(object,label,colour,last_n=NULL,ncol=2L,title=NULL,...){
  oos<-.get_oos(object); hist_df<-if(!is.null(last_n)) tail(object$data,last_n) else object$data
  ev<-if(!is.null(object$cv_results))"modeltime.resample" else "Hold-out"
  .oos_dash(oos,hist_df,title%||%sprintf("%s — OOS (v6)",label),ev,label,ncol)
}
#' @export
plot_oos_fit.auto_arima_ts  <- function(object,...) .make_oos_plot(object,sprintf("Auto-ARIMA(%d,%d,%d)",object$model$arma[1],object$model$arma[6],object$model$arma[2]),.PAL$arima,...)
#' @export
plot_oos_fit.auto_ets_ts    <- function(object,...) .make_oos_plot(object,sprintf("Auto-ETS(%s)",object$model$method),.PAL$ets,...)
#' @export
plot_oos_fit.auto_arfima_ts <- function(object,...) .make_oos_plot(object,sprintf("Auto-ARFIMA(d=%.4f)",object$d_hat),.PAL$arfima,...)
#' @export
plot_oos_fit.auto_har_rv    <- function(object,...) .make_oos_plot(object,sprintf("HAR-RV[%s]",object$variant),.PAL$har,...)
#' @export
plot_oos_fit.auto_garch     <- function(object,...) .make_oos_plot(object,sprintf("GARCH[%s(%d,%d)]",object$model_type,object$p,object$q),.PAL$garch,...)
#' @export
plot_oos_fit.default <- function(object,hist_df=NULL,title="OOS Actual vs Fitted",ncol=2L,...){
  if(!"actual"%in%names(object)) stop("Need columns: actual, predicted (or fitted).")
  .oos_dash(object,hist_df,title,"modeltime.resample","Model",ncol)
}

#' modeltime-style forecast plot
#' @export
plot_tf_forecast <- function(combined_tbl,...){
  pal_m <- setNames(scales::hue_pal()(dplyr::n_distinct(combined_tbl$.model_desc)),
                    unique(combined_tbl$.model_desc))
  pal_m["Actual"] <- .PAL$actual
  p <- ggplot(combined_tbl,aes(date,.value,colour=.model_desc))+
    geom_line(data=filter(combined_tbl,.key=="actual"),colour=.PAL$actual,linewidth=0.9)+
    geom_ribbon(data=filter(combined_tbl,.key=="prediction"&!is.na(.conf_lo)),
                aes(ymin=.conf_lo,ymax=.conf_hi,fill=.model_desc),alpha=0.12,colour=NA)+
    geom_line(data=filter(combined_tbl,.key=="prediction"),linewidth=0.95,linetype="dashed")+
    geom_point(data=filter(combined_tbl,.key=="prediction"),size=2,shape=21,fill="white",stroke=1)+
    scale_colour_manual(values=pal_m,name="Model")+scale_fill_manual(values=pal_m,guide="none")+
    labs(title="torchforecast v6 — Forecast (modeltime-style)",
         subtitle="Solid=actual | Dashed=prediction | Band=CI",x=NULL,y="Value")+.bt()
  print(p); invisible(p)
}


# ================================================================
# ██  PART 21 — DIEBOLD-MARIANO TEST  (unchanged)
# ================================================================

#' @export
dm_test <- function(e1,e2,h=1L,
                    loss=c("MSE","MAE","MAPE","SMAPE","MBE","POWER","LINEX","custom"),
                    power=1,linex_a=1,actual=NULL,custom_fn=NULL,
                    alternative=c("two.sided","less","greater"),
                    hln=TRUE,bw_method=c("fixed","andrews","nw_auto"),conf_level=0.95){
  loss<-match.arg(loss);alternative<-match.arg(alternative);bw_method<-match.arg(bw_method)
  e1<-as.numeric(e1);e2<-as.numeric(e2)
  if(length(e1)!=length(e2)) stop("e1 and e2 must have equal length.")
  na_mask<-is.na(e1)|is.na(e2)
  if(!is.null(actual)){actual<-as.numeric(actual);na_mask<-na_mask|is.na(actual);actual<-actual[!na_mask]}
  e1<-e1[!na_mask];e2<-e2[!na_mask];T<-length(e1)
  if(T<10L) warning("DM test: T<10.")
  .L<-function(e,a) switch(loss,MSE=e^2,MAE=abs(e),
    MAPE={if(is.null(a)) stop("actual required"); abs(e/a)*100},
    SMAPE={if(is.null(a)) stop("actual required"); 200*abs(e)/(abs(a)+abs(a-e))},
    MBE=e,POWER=abs(e)^power,LINEX=exp(linex_a*e)-linex_a*e-1,
    custom={if(is.null(custom_fn)) stop("custom_fn required"); custom_fn(e,a)})
  d<-.L(e1,actual)-.L(e2,actual); d_bar<-mean(d); lm_d<-lm(d~1)
  bw<-switch(bw_method,
    fixed=max(0L,h-1L),
    andrews=tryCatch(sandwich::bwAndrews(lm_d),error=function(e) max(0L,h-1L)),
    nw_auto=tryCatch(sandwich::bwNeweyWest(lm_d),error=function(e) max(0L,h-1L)))
  vcov_hac<-sandwich::NeweyWest(lm_d,lag=max(0L,bw),prewhite=FALSE,adjust=TRUE)
  lrv<-as.numeric(vcov_hac); se<-sqrt(lrv/T)
  if(se<.Machine$double.eps) stop("SE is zero.")
  DM<-d_bar/se; hln_f<-1; df_stat<-Inf
  if(hln){hln_f<-sqrt((T+1-2*h+h*(h-1)/T)/T); df_stat<-T-1L}
  DM_adj<-DM*hln_f
  p_val<-switch(alternative,
    two.sided=2*(if(is.finite(df_stat)) pt(-abs(DM_adj),df=df_stat) else pnorm(-abs(DM_adj))),
    less     =  (if(is.finite(df_stat)) pt(DM_adj,df=df_stat)  else pnorm(DM_adj)),
    greater  =  (if(is.finite(df_stat)) pt(-DM_adj,df=df_stat) else pnorm(-DM_adj)))
  alpha<-1-conf_level; q<-if(is.finite(df_stat)) qt(1-alpha/2,df=df_stat) else qnorm(1-alpha/2)
  ci<-c(lower=d_bar-q*se,upper=d_bar+q*se)
  concl<-if(p_val>=(1-conf_level))
    sprintf("FAIL TO REJECT H0 (p=%.4f): No significant difference.",p_val)
  else sprintf("REJECT H0 (p=%.4f): %s %s",p_val,
    if(d_bar>0)"Model 1 has HIGHER loss" else "Model 1 has LOWER loss",
    dplyr::case_when(p_val<0.001~"***",p_val<0.01~"**",p_val<0.05~"*",TRUE~""))
  structure(list(statistic=DM_adj,dm_raw=DM,p_value=p_val,alternative=alternative,loss=loss,
    h=h,T=T,d_bar=d_bar,lrv=lrv*T,se=se,ci=ci,conf_level=conf_level,
    hln=hln,hln_factor=hln_f,df=df_stat,bw=bw,bw_method=bw_method,
    d_series=d,L1=.L(e1,actual),L2=.L(e2,actual),conclusion=concl),class="dm_test")
}
#' @export
print.dm_test <- function(x,...){
  cat("\n",.box(),"  Diebold-Mariano Test\n",.box(),"\n",sep="")
  cat(sprintf("  Loss=%-6s|h=%d|n=%d|alt=%s|HLN=%s\n",x$loss,x$h,x$T,x$alternative,x$hln))
  cat(sprintf("  d̄=%+.6f|SE=%.6f|DM*=%+.4f|p=%.6f %s\n",x$d_bar,x$se,x$statistic,x$p_value,
    dplyr::case_when(x$p_value<0.001~"***",x$p_value<0.01~"**",x$p_value<0.05~"*",x$p_value<0.10~".",TRUE~"")))
  cat(sprintf("  %.0f%% CI: [%+.6f, %+.6f]\n",x$conf_level*100,x$ci["lower"],x$ci["upper"]))
  cat(sprintf("  %s\n",x$conclusion)); cat(.box(),"\n\n"); invisible(x)
}

#' DM test from two torchforecast objects
#' @export
dm_compare <- function(result1,result2,strategy1="recursive",strategy2="recursive",
                        loss="MSE",h=NULL,by_horizon=TRUE,alternative="two.sided",hln=TRUE,...){
  oos1<-.get_oos(result1,strategy1); oos2<-.get_oos(result2,strategy2)
  if("strategy"%in%names(oos1)) oos1<-dplyr::filter(oos1,strategy==strategy1)
  if("strategy"%in%names(oos2)) oos2<-dplyr::filter(oos2,strategy==strategy2)
  pc<-function(df) if("fitted"%in%names(df))"fitted" else "predicted"
  n_use<-min(nrow(oos1),nrow(oos2))
  e1<-(oos1$actual-oos1[[pc(oos1)]])[seq_len(n_use)]
  e2<-(oos2$actual-oos2[[pc(oos2)]])[seq_len(n_use)]
  act<-oos1$actual[seq_len(n_use)]
  dm_res<-dm_test(e1,e2,h=h%||%1L,loss=loss,actual=act,alternative=alternative,hln=hln,...)
  dm_h<-if(by_horizon&&is.null(h)) tryCatch({
    H<-max(oos1$h%||%1L,oos2$h%||%1L)
    rows<-lapply(seq_len(H),function(hh){
      d1<-dplyr::filter(oos1,h==hh); d2<-dplyr::filter(oos2,h==hh)
      if(nrow(d1)<5||nrow(d2)<5) return(NULL)
      n<-min(nrow(d1),nrow(d2))
      res<-tryCatch(dm_test(d1$actual[1:n]-d1[[pc(d1)]][1:n],d2$actual[1:n]-d2[[pc(d2)]][1:n],
                             h=hh,loss=loss,actual=d1$actual[1:n],alternative=alternative,hln=hln,...),
                    error=function(e) NULL)
      if(is.null(res)) return(NULL)
      tibble(h=hh,n=n,DM_stat=res$statistic,p_value=res$p_value,d_bar=res$d_bar,
             sig=dplyr::case_when(res$p_value<0.01~"***",res$p_value<0.05~"**",
                                  res$p_value<0.10~"*",TRUE~""))
    }); dplyr::bind_rows(Filter(Negate(is.null),rows))
  },error=function(e) NULL) else NULL
  list(dm=dm_res,dm_horizon=dm_h,oos1=oos1,oos2=oos2)
}


# ================================================================
# ██  PART 22 — compare_all_models()
# ================================================================

#' @export
compare_all_models <- function(data, rnn_models=c("lstm","gru","rnn"),
                                run_nbeats=FALSE, run_arima=TRUE, run_arfima=TRUE,
                                run_ets=FALSE, run_har=FALSE, run_garch=FALSE,
                                horizon=12L, rank_by="RMSE", rnn_strategy="recursive",
                                run_cv=TRUE, cv_holdout_frac=0.20,
                                cv_initial=NULL, cv_window=NULL, cv_skip=1L,
                                rnn_config_obj=NULL, use_parallel=TRUE,
                                har_rv_col="value", garch_model="sGARCH",...){
  total_t0 <- .tic(); use_par <- use_parallel&&.is_parallel()
  all_names <- c(toupper(rnn_models%||%character(0)),
                 if(run_nbeats)"NBEATS",if(run_arima)"ARIMA",
                 if(run_arfima)"ARFIMA",if(run_ets)"ETS",
                 if(run_har)"HAR",if(run_garch)"GARCH")
  cat(sprintf("\n── %s comparison | %d models | workers=%d\n",
    if(use_par)"Parallel" else "Sequential",length(all_names),.n_workers()))
  cat(sprintf("   CV: modeltime.resample | BO: ParBayesianOptimization\n"))
  cat(sprintf("   Fast numerics: matrixStats + RcppRoll\n"))

  train_model <- function(nm, dat){
    tryCatch(switch(nm,
      LSTM=auto_lstm(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                     cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,
                     config=rnn_config_obj,use_parallel=FALSE,verbose=0L,...),
      GRU=auto_gru(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                    cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,
                    config=rnn_config_obj,use_parallel=FALSE,verbose=0L,...),
      RNN=auto_rnn(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                    cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,
                    config=rnn_config_obj,use_parallel=FALSE,verbose=0L,...),
      NBEATS=auto_nbeats(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                          cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,
                          config=rnn_config_obj,use_parallel=FALSE,verbose=0L,...),
      ARIMA=auto_arima_ts(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                           cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,verbose=0L),
      ARFIMA=auto_arfima_ts(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                             cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,verbose=0L),
      ETS=auto_ets_ts(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                       cv_initial=cv_initial,cv_window=cv_window,cv_skip=cv_skip,verbose=0L),
      HAR=auto_har_rv(dat,rv_col=har_rv_col,horizon=min(horizon,5L),run_cv=run_cv,
                       cv_holdout_frac=cv_holdout_frac,cv_initial=cv_initial,cv_window=cv_window,
                       cv_skip=cv_skip,verbose=0L),
      GARCH=auto_garch(dat,horizon=horizon,model=garch_model,run_cv=run_cv,
                        cv_holdout_frac=cv_holdout_frac,cv_initial=cv_initial,cv_window=cv_window,
                        cv_skip=cv_skip,verbose=0L),
      stop(sprintf("Unknown: %s",nm))
    ),error=function(e){message(sprintf("  [%s] FAILED: %s",nm,e$message));NULL})
  }

  if(use_par){
    results_list <- furrr::future_map(all_names,
      function(nm){suppressPackageStartupMessages({library(torch);library(rugarch)});train_model(nm,data)},
      .options=furrr::furrr_options(seed=TRUE,globals=FALSE),.progress=TRUE)
    names(results_list) <- all_names
  } else {
    results_list <- lapply(setNames(all_names,all_names),function(nm){
      cat(sprintf("\n── Training %s ──\n",nm)); train_model(nm,data)})
  }
  results <- Filter(Negate(is.null),results_list)

  .pull_m <- function(nm, res, metric){
    if(is.null(res)) return(NA_real_)
    if(inherits(res,"auto_rnn")){
      strat <- if(res$model_type=="nbeats")"nbeats" else rnn_strategy
      if(!is.null(res$cv_results)&&!is.null(res$cv_results$cv_metrics)){
        m <- res$cv_results$cv_metrics
        if("strategy"%in%names(m)){r<-m[m$strategy==strat,,drop=FALSE];
          if(nrow(r)>0&&metric%in%names(r)) return(r[[metric]][1])}
        else if(metric%in%names(m)) return(m[[metric]][1])
      }
      if(!is.null(res$holdout_results)){
        m <- res$holdout_results[[paste0("metrics_",strat)]]
        if(!is.null(m)) return(m[[metric]])
      }
    } else {
      if(!is.null(res$cv_results)&&!is.null(res$cv_results$cv_metrics)){
        m <- res$cv_results$cv_metrics
        if(metric%in%names(m)) return(m[[metric]][1])
      }
      if(!is.null(res$holdout_results)){
        ho <- res$holdout_results
        pc <- if("fitted"%in%names(ho))"fitted" else "predicted"
        if(!is.null(ho)&&pc%in%names(ho)){
          r <- ho$actual-ho[[pc]]
          return(switch(metric,RMSE=sqrt(mean(r^2,na.rm=TRUE)),MAE=mean(abs(r),na.rm=TRUE),
            MAPE=mean(abs(r/ho$actual)*100,na.rm=TRUE),
            SMAPE=mean(200*abs(r)/(abs(ho$actual)+abs(ho[[pc]])),na.rm=TRUE),NA_real_))
        }
      }
    }
    NA_real_
  }

  metrics_show <- c("RMSE","MAE","MAPE","SMAPE")
  ranking <- dplyr::bind_rows(lapply(names(results),function(nm){
    row <- tibble(model=nm)
    for(m in metrics_show) row[[m]] <- .pull_m(nm,results[[nm]],m)
    row$runtime_sec <- {rt<-results[[nm]]$runtime;if(is.null(rt)) NA_real_ else rt$total_sec%||%NA_real_}
    row$runtime_fmt <- .fmt_elapsed(row$runtime_sec); row
  })) %>% arrange(.data[[rank_by]])

  total_sec <- .toc(total_t0)
  cat("\n",.box(),"\n  Model Comparison (v6)\n",.box(),"\n",sep="")
  cat(sprintf("  CV: modeltime.resample | BO: ParBayesianOptimization\n"))
  cat(sprintf("  Ranked by %s | total=%s\n",rank_by,.fmt_elapsed(total_sec)))
  cat(sprintf("  %-10s","Model"))
  for(m in metrics_show) cat(sprintf(" %10s",m))
  cat(sprintf(" %12s\n","Runtime")); cat(" ",strrep("-",64),"\n",sep="")
  for(i in seq_len(nrow(ranking))){
    cat(sprintf("  %-10s",ranking$model[i]))
    for(m in metrics_show) cat(sprintf(" %10s",.fmt(ranking[[m]][i],4)))
    cat(sprintf(" %12s",ranking$runtime_fmt[i]))
    if(i==1L) cat("  ← best"); cat("\n")
  }
  cat(.box(),"\n\n")

  model_pal <- unlist(.PAL[c("LSTM","GRU","RNN","NBEATS","ARIMA","ARFIMA","ETS","HAR","GARCH")])
  bt <- .bt()+theme(plot.title=element_text(face="bold",size=13,hjust=0.5),
                    plot.subtitle=element_text(hjust=0.5,colour="grey40"),
                    axis.text.x=element_text(angle=30,hjust=1))
  plot_long <- ranking%>%select(model,all_of(metrics_show))%>%
    pivot_longer(-model,names_to="metric",values_to="value")%>%filter(!is.na(value))%>%
    mutate(metric=factor(metric,levels=metrics_show),model=factor(model,levels=ranking$model))
  comp_plot <- ggplot(plot_long,aes(model,value,fill=model))+
    geom_col(width=0.6,show.legend=FALSE)+geom_text(aes(label=round(value,3)),vjust=-0.4,size=3)+
    facet_wrap(~metric,scales="free_y",nrow=1L)+
    scale_fill_manual(values=model_pal,breaks=names(model_pal))+
    labs(title="Model Comparison — OOS Metrics (modeltime.resample)",
         subtitle=sprintf("Ranked by %s | total=%s",rank_by,.fmt_elapsed(total_sec)),x=NULL,y="Value")+bt
  print(comp_plot)
  rt_plot <- ranking%>%filter(!is.na(runtime_sec))%>%mutate(model=factor(model,levels=model))%>%
    ggplot(aes(model,runtime_sec,fill=model))+geom_col(width=0.55,show.legend=FALSE)+
    geom_text(aes(label=runtime_fmt),vjust=-0.4,size=3)+
    scale_fill_manual(values=model_pal,breaks=names(model_pal))+
    labs(title="Runtime | BO: ParBayesianOptimization",x=NULL,y="Seconds")+bt
  print(rt_plot)
  list(ranking=ranking,results=results,plot=comp_plot,runtime_plot=rt_plot,total_sec=total_sec)
}


# ================================================================
# ██  PART 23 — EXAMPLE USAGE
# ================================================================

# source("torchforecast.R"); library(zoo)
#
# df <- data.frame(
#   date  = seq(as.Date("1949-01-01"), by="month", length.out=144),
#   value = as.numeric(AirPassengers))
# train_df <- df[1:120,]; test_df <- df[121:144,]
#
# ── 1. Parallel setup (registers doFuture for ParBayesianOptimization)
# tf_setup_parallel()          # auto-detect cores; also calls doFuture::registerDoFuture()
# tf_setup_parallel(workers=4L)
#
# ── 2. Configuration
# cfg <- rnn_config(
#   run_bo=TRUE, bo_iter=12L,
#   bo_acq="ucb", bo_kappa=2.576,          # ParBayesianOptimization acq args
#   run_cv=TRUE,
#   cv_window=NULL, cv_skip=1L,            # expanding window, skip 1 between folds
#   # cv_window=84L                        # OR fixed sliding window of 84 obs
#   seed=42L
# )
#
# ── 3. Train individual models
# lstm  <- auto_lstm(train_df, horizon=24L, config=cfg)
# arima <- auto_arima_ts(train_df, horizon=24L, run_cv=TRUE, cv_skip=1L)
# ets   <- auto_ets_ts(train_df,   horizon=24L, run_cv=TRUE, cv_skip=2L)
#
# ── 4. matrixStats / RcppRoll usage (internal — called automatically)
# # make_supervised()   → vectorised lag matrix (no base R loop)
# # scale_matrix_cols() → matrixStats::colMeans2 / colSds
# # .roll_mean_right()  → RcppRoll::roll_mean (HAR-RV features)
#
# ── 5. predict() — three signatures
# predict(lstm,  horizon=24L)             # Sig 1: future
# predict(lstm,  new_data=test_df)        # Sig 2: rolling OOS on test set
# predict(arima, new_data=test_df, horizon=12L)  # Sig 3: forecast from test tail
#
# ── 6. modeltime.resample accuracy
# resample_accuracy(lstm)                 # thin wrapper → same as cv_performance
# cv_performance(lstm)
# cv_performance(arima)
#
# ── 7. tf_calibrate (modeltime-style)
# cal <- tf_calibrate(list(LSTM=lstm, ARIMA=arima, ETS=ets), test_df)
# tf_accuracy(cal)                        # RMSE/MAE/MAPE/SMAPE/RSQ table
# fc  <- tf_forecast(cal, horizon=24L)
# combined <- tf_forecast_combined(cal, horizon=24L)
# plot_tf_forecast(combined)
#
# ── 8. modeltime table
# mt <- tf_modeltime_table(lstm, arima, ets)
#
# ── 9. Compare all
# comp <- compare_all_models(
#   train_df, rnn_models=c("lstm","gru"),
#   run_nbeats=TRUE, run_arima=TRUE, run_ets=TRUE,
#   horizon=24L, run_cv=TRUE,
#   cv_window=NULL, cv_skip=1L,       # modeltime.resample expanding
#   rnn_config_obj=rnn_config(bo_iter=8L),
#   use_parallel=TRUE
# )
# predict(comp$results[[comp$ranking$model[1]]], new_data=test_df)  # best model
#
# ── 10. DM test
# dm_out <- dm_compare(lstm, arima)
# print(dm_out$dm)
#
# tf_reset_parallel()
