# ================================================================
# torchforecast.R  —  Unified Time Series Forecasting Framework
# Version 4.0  (single file)
# ================================================================
#
# KEY CHANGE v4.0: All rolling-origin CV now uses forecast::tsCV()
# as the unified engine for every model type (LSTM, GRU, RNN,
# N-BEATS, ARIMA, ARFIMA, HAR-RV, GARCH).
#
# forecast::tsCV(y, forecastfunction, h, initial, window)
#   y                — ts object of the full series
#   forecastfunction — function(y, h, ...) returning forecast object
#   h                — forecast horizon
#   initial          — minimum training period
#   window           — fixed window size (NULL = expanding)
#
# tsCV returns an error matrix [T × h] where entry (t, h) is the
# h-step-ahead forecast error at time t. NAs fill positions where
# there is insufficient history. All per-horizon metrics (RMSE,
# MAE, MAPE, SMAPE, MBE) are computed from these error matrices.
#
# MODELS
# ────────────────────────────────────────────────────────────────
#   Deep Learning (native R torch, no Python)
#     LSTM  · GRU  · Vanilla RNN
#     N-BEATS (Neural Basis Expansion Analysis)
#
#   Statistical / Econometric
#     Auto-ARIMA   · Auto-ARFIMA · HAR-RV · GARCH
#
# INSTALLATION (run once)
# ────────────────────────────────────────────────────────────────
#   install.packages(c(
#     "torch","tidyverse","rsample","zoo","rBayesianOptimization",
#     "forecast","fracdiff","tseries","rugarch","patchwork",
#     "scales","sandwich","lmtest","future","furrr","parallelly"
#   ))
#   torch::install_torch()
#
# USAGE
# ────────────────────────────────────────────────────────────────
#   source("torchforecast.R")
#   tf_setup_parallel()           # optional
#   result <- auto_lstm(df, horizon = 12L)
#   print(result); plot(result); plot_oos_fit(result)
# ================================================================

suppressPackageStartupMessages({
  library(torch);    library(tidyverse); library(rsample)
  library(zoo);      library(rBayesianOptimization)
  library(forecast); library(fracdiff);  library(tseries)
  library(rugarch);  library(patchwork); library(scales)
  library(sandwich); library(lmtest)
  library(future);   library(furrr);     library(parallelly)
})


# ================================================================
# ██  PART 0 — GLOBAL CONSTANTS & HELPERS
# ================================================================

.VALID_RNN <- c("lstm","gru","rnn")
.PAL <- list(
  recursive="#2166ac", direct="#d6604d", nbeats="#1d91c0",
  arima="#1b7837",     arfima="#8c510a", har="#e08214",
  garch="#762a83",     actual="grey20",  neutral="grey60",
  train="#35978f",     val="#762a83",
  LSTM="#2166ac", GRU="#d6604d", RNN="#984ea3", NBEATS="#1d91c0",
  ARIMA="#1b7837", ARFIMA="#8c510a", HAR="#e08214", GARCH="#762a83"
)

`%||%`         <- function(a,b) if(!is.null(a)) a else b
.fmt           <- function(x,d=4) formatC(x,digits=d,format="f",flag=" ")
.box           <- function(ch="═",w=66) strrep(ch,w)
.hdr           <- function(txt,ch="─",w=66) {
  pad <- max(0L,w-nchar(txt)-4L)
  sprintf("%s  %s  %s",strrep(ch,2),txt,strrep(ch,pad))
}
.tic           <- function() proc.time()["elapsed"]
.toc           <- function(t0) as.numeric(proc.time()["elapsed"]-t0)
.fmt_elapsed   <- function(s) {
  if(is.na(s)||is.null(s)) return("NA")
  if(s<60)    sprintf("%.1fs",s)
  else if(s<3600) sprintf("%.1fmin",s/60)
  else sprintf("%.2fh",s/3600)
}

make_scaler    <- function(x) list(mean=mean(x,na.rm=TRUE),sd=sd(x,na.rm=TRUE))
scale_x        <- function(x,sc) (x-sc$mean)/sc$sd
unscale_x      <- function(x,sc)  x*sc$sd+sc$mean
get_device     <- function() {
  if(cuda_is_available()) "cuda"
  else if(backends_mps_is_available()) "mps"
  else "cpu"
}
infer_frequency <- function(dates) {
  med <- median(as.numeric(diff(sort(dates))))
  dplyr::case_when(med<=1~"day",med<=8~"week",med<=32~"month",med<=93~"quarter",TRUE~"year")
}
.ts_freq <- function(dates) {
  med <- median(as.numeric(diff(sort(dates))))
  dplyr::case_when(med<=1~365L,med<=8~52L,med<=32~12L,med<=93~4L,TRUE~1L)
}
.df_to_ts <- function(df) {
  f <- .ts_freq(df$date)
  s <- c(as.integer(format(min(df$date),"%Y")),
         if(f==12L) as.integer(format(min(df$date),"%m"))
         else if(f==4L) as.integer(ceiling(as.integer(format(min(df$date),"%m"))/3))
         else 1L)
  ts(df$value,start=s,frequency=f)
}
make_supervised <- function(x,lags) {
  n <- length(x); nr <- n-lags
  X <- matrix(NA_real_,nr,lags); y <- numeric(nr)
  for(i in seq_len(nr)){X[i,]<-x[i:(i+lags-1)];y[i]<-x[i+lags]}
  list(X=X,y=y)
}
make_supervised_direct <- function(x,lags,h) {
  n <- length(x); nr <- n-lags-h+1L
  if(nr<=0L) stop("Not enough data for direct strategy.")
  X <- matrix(NA_real_,nr,lags); y <- numeric(nr)
  for(i in seq_len(nr)){X[i,]<-x[i:(i+lags-1)];y[i]<-x[i+lags+h-1L]}
  list(X=X,y=y)
}
.safe_metrics <- function(actual,predicted) {
  r <- actual-predicted
  list(
    RMSE=sqrt(mean(r^2,na.rm=TRUE)),
    MAE=mean(abs(r),na.rm=TRUE),
    MAPE=if(any(actual==0,na.rm=TRUE)) NA_real_
         else mean(abs(r/actual)*100,na.rm=TRUE),
    SMAPE=mean(200*abs(r)/(abs(actual)+abs(predicted)),na.rm=TRUE),
    MBE=mean(r,na.rm=TRUE)
  )
}
.check_arch <- function(arch) {
  al <- tolower(trimws(arch))
  valid <- c(.VALID_RNN,"nbeats")
  if(!al %in% valid) stop(sprintf("arch must be one of: %s",paste(valid,collapse=", ")))
  al
}
.base_theme <- function()
  theme_minimal(base_size=11)+
  theme(plot.title=element_text(face="bold",size=11),
        plot.subtitle=element_text(size=9,colour="grey45"),
        legend.position="top",legend.key.size=unit(0.4,"cm"))


# ================================================================
# ██  PART 1 — PARALLEL BACKEND
# ================================================================

#' Configure parallel backend for torchforecast
#' @export
tf_setup_parallel <- function(workers=NULL,
                               backend=c("multisession","multicore","cluster","sequential"),
                               gc=TRUE, verbose=TRUE) {
  backend <- match.arg(backend)
  workers <- if(is.null(workers)||identical(workers,"auto"))
    max(1L,parallelly::availableCores(which="system")-1L)
  else if(identical(workers,"max")) parallelly::availableCores()
  else { w <- as.integer(workers); a <- parallelly::availableCores()
         if(w>a){warning(sprintf("Only %d cores available.",a));a} else w }
  if(backend=="multicore"){
    if(.Platform$OS.type=="windows"){warning("→ multisession");backend<-"multisession"}
    else if(isNamespaceLoaded("torch")){message("torch loaded → multisession");backend<-"multisession"}
  }
  if(backend=="sequential"||workers<=1L){
    future::plan(future::sequential); workers <- 1L
  } else {
    future::plan(switch(backend,
      multisession=future::multisession,
      multicore=future::multicore,
      cluster=future::cluster),
      workers=workers,gc=gc)
  }
  options(tf.workers=workers,tf.backend=backend)
  if(verbose){
    cat(sprintf("\n── torchforecast parallel ──────────────────────────────────\n"))
    cat(sprintf("   Backend: %-16s  Workers: %d\n",backend,workers))
    cat(strrep("─",60),"\n\n")
  }
  invisible(workers)
}

#' Reset to sequential execution
#' @export
tf_reset_parallel <- function() {
  future::plan(future::sequential)
  options(tf.workers=1L,tf.backend="sequential")
  message("torchforecast: sequential."); invisible(NULL)
}
.n_workers   <- function() getOption("tf.workers",default=1L)
.is_parallel <- function() .n_workers()>1L


# ================================================================
# ██  PART 2 — rnn_config()
# ================================================================

#' @export
rnn_config <- function(
    hidden_size=64L,num_layers=2L,dropout=0.1,dense_units=32L,
    rnn_nonlinearity="tanh",
    lr=1e-3,batch_size=32L,final_epochs=200L,final_patience=25L,
    lr_factor=0.5,lr_patience=NULL,grad_clip=1.0,val_fraction=0.15,
    run_bo=TRUE,bo_init=5L,bo_iter=15L,bo_epochs=80L,bo_patience=12L,
    bo_acq="ucb",bo_kappa=2.576,bo_bounds=NULL,
    run_cv=TRUE,cv_initial=NULL,cv_window=NULL,
    cv_holdout_frac=0.20,
    lags=NULL,scale=TRUE,seed=42L,verbose=1L) {
  # cv_window: NULL = expanding window; integer = fixed sliding window (passed to tsCV)
  .vbn <- c("hidden_size","num_layers","dropout","dense_units","lr_log10","batch_size_log2")
  if(!is.null(bo_bounds)){
    bad <- setdiff(names(bo_bounds),.vbn)
    if(length(bad)) stop(sprintf("Unknown bo_bounds: %s",paste(bad,collapse=",")))
    for(nm in names(bo_bounds)){b<-bo_bounds[[nm]]
      if(!is.numeric(b)||length(b)!=2||b[1]>=b[2])
        stop(sprintf("bo_bounds$%s: need c(lo,hi) with lo<hi",nm))}
  }
  structure(list(
    hidden_size=as.integer(hidden_size),num_layers=as.integer(num_layers),
    dropout=dropout,dense_units=as.integer(dense_units),
    rnn_nonlinearity=rnn_nonlinearity,
    lr=lr,batch_size=as.integer(batch_size),
    final_epochs=as.integer(final_epochs),final_patience=as.integer(final_patience),
    lr_factor=lr_factor,lr_patience=lr_patience,grad_clip=grad_clip,
    val_fraction=val_fraction,
    run_bo=run_bo,bo_init=as.integer(bo_init),bo_iter=as.integer(bo_iter),
    bo_epochs=as.integer(bo_epochs),bo_patience=as.integer(bo_patience),
    bo_acq=bo_acq,bo_kappa=bo_kappa,bo_bounds=bo_bounds,
    run_cv=run_cv,cv_initial=cv_initial,cv_window=cv_window,
    cv_holdout_frac=cv_holdout_frac,
    lags=lags,scale=scale,seed=as.integer(seed),verbose=as.integer(verbose)
  ),class="rnn_config")
}
#' @export
print.rnn_config <- function(x,...) {
  cat("\n",.box(),"\n  rnn_config\n",.box(),"\n",sep="")
  cat(sprintf("  Arch  : hidden=%d|layers=%d|dropout=%.2f|dense=%d\n",
              x$hidden_size,x$num_layers,x$dropout,x$dense_units))
  cat(sprintf("  Train : lr=%.2e|batch=%d|epochs=%d|patience=%d\n",
              x$lr,x$batch_size,x$final_epochs,x$final_patience))
  cat(sprintf("  BO    : %s",if(x$run_bo)"ON" else "OFF"))
  if(x$run_bo) cat(sprintf(" (init=%d|iter=%d|acq=%s)",x$bo_init,x$bo_iter,x$bo_acq))
  cat("\n")
  cat(sprintf("  CV    : %s | engine=forecast::tsCV%s\n",
    if(x$run_cv)"rolling-origin" else "hold-out",
    if(!is.null(x$cv_window)) sprintf(" [window=%d]",x$cv_window) else " [expanding]"))
  invisible(x)
}
.resolve_cfg <- function(config,overrides) {
  base <- rnn_config()
  if(!is.null(config)){
    if(!inherits(config,"rnn_config")) stop("config must be from rnn_config().")
    for(nm in names(config)) base[[nm]] <- config[[nm]]
  }
  for(nm in names(overrides)) if(!is.null(overrides[[nm]])) base[[nm]] <- overrides[[nm]]
  base
}
.default_bo_bounds <- list(
  hidden_size=c(16L,128L),num_layers=c(1L,3L),dropout=c(0.0,0.4),
  dense_units=c(16L,128L),lr_log10=c(-4.0,-2.0),batch_size_log2=c(3.0,6.0))
.resolve_bounds <- function(user) {
  b <- .default_bo_bounds
  if(!is.null(user)) for(nm in names(user)) b[[nm]] <- user[[nm]]
  b
}


# ================================================================
# ██  PART 3 — TORCH DATASET
# ================================================================

TimeSeriesDataset <- dataset(
  name="TimeSeriesDataset",
  initialize=function(X_mat,y_vec){
    self$X <- torch_tensor(X_mat,dtype=torch_float())$unsqueeze(3L)
    self$y <- torch_tensor(as.numeric(y_vec),dtype=torch_float())$unsqueeze(2L)
  },
  .getitem=function(i) list(x=self$X[i,,],y=self$y[i,]),
  .length=function() dim(self$X)[1L]
)


# ================================================================
# ██  PART 4 — STACKEDRNN (LSTM / GRU / RNN)
# ================================================================

StackedRNN <- nn_module(
  classname="StackedRNN",
  initialize=function(model_type="lstm",input_size=1L,hidden_size=64L,
                      num_layers=2L,dropout=0.1,dense_units=32L,
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
    self$fc1   <- nn_linear(hidden_size,dense_units)
    self$relu  <- nn_relu()
    self$drop2 <- nn_dropout(p=dropout/2)
    self$out   <- nn_linear(dense_units,1L)
  },
  forward=function(x){
    o <- self$rnn(x)[[1L]]
    last <- o[,dim(o)[2L],]
    last |> self$drop1() |> self$fc1() |> self$relu() |> self$drop2() |> self$out()
  }
)


# ================================================================
# ██  PART 5 — N-BEATS
# ================================================================

NBEATSBlock <- nn_module(
  classname="NBEATSBlock",
  initialize=function(input_size,theta_size,fc_width=512L,n_layers=4L,
                      basis_type="generic",horizon=12L){
    self$basis_type <- basis_type
    self$theta_size <- theta_size
    self$input_size <- input_size
    self$horizon    <- horizon
    layers <- list(); in_sz <- input_size
    for(i in seq_len(n_layers)){
      layers[[2L*i-1L]] <- nn_linear(in_sz,fc_width)
      layers[[2L*i]]    <- nn_relu()
      in_sz <- fc_width
    }
    self$fc_stack  <- nn_sequential(!!!layers)
    self$theta_b   <- nn_linear(fc_width,theta_size,bias=FALSE)
    self$theta_f   <- nn_linear(fc_width,theta_size,bias=FALSE)
    if(basis_type=="generic"){
      self$backcast_basis <- nn_linear(theta_size,input_size,bias=FALSE)
      self$forecast_basis <- nn_linear(theta_size,horizon,    bias=FALSE)
    }
  },
  .trend_basis=function(T,degree,dev="cpu"){
    t    <- torch_arange(0,T-1L,dtype=torch_float(),device=dev)$unsqueeze(2L)/(T-1L)
    pows <- torch_arange(0,degree-1L,dtype=torch_float(),device=dev)$unsqueeze(1L)
    t$pow(pows)
  },
  .seasonality_basis=function(T,n_harmonics,dev="cpu"){
    t <- torch_arange(0,T-1L,dtype=torch_float(),device=dev)
    cols <- list()
    for(i in seq_len(n_harmonics)){
      arg <- 2*pi*i*t/T
      cols[[2L*i-1L]] <- arg$cos()
      cols[[2L*i]]    <- arg$sin()
    }
    torch_stack(cols,dim=2L)
  },
  forward=function(x){
    dev <- x$device; h <- self$fc_stack(x)
    tb  <- self$theta_b(h); tf_ <- self$theta_f(h)
    if(self$basis_type=="generic"){
      backcast <- self$backcast_basis(tb)
      forecast <- self$forecast_basis(tf_)
    } else {
      deg  <- self$theta_size%/%2L; n_h <- self$theta_size-deg
      T_b  <- self$input_size;      T_f <- self$horizon
      Vb   <- self$.trend_basis(T_b,deg,dev$type)
      Vf   <- self$.trend_basis(T_f,deg,dev$type)
      backcast <- torch_matmul(tb[,1:deg],Vb$t())
      forecast <- torch_matmul(tf_[,1:deg],Vf$t())
      if(n_h>0L){
        n_harm <- n_h%/%2L
        Sb <- self$.seasonality_basis(T_b,n_harm,dev$type)
        Sf <- self$.seasonality_basis(T_f,n_harm,dev$type)
        backcast <- backcast+torch_matmul(tb[,(deg+1):self$theta_size],Sb$t())
        forecast <- forecast+torch_matmul(tf_[,(deg+1):self$theta_size],Sf$t())
      }
    }
    list(backcast=backcast,forecast=forecast)
  }
)

NBEATSModel <- nn_module(
  classname="NBEATSModel",
  initialize=function(input_size,horizon,n_stacks=2L,n_blocks=3L,
                      fc_width=512L,n_layers=4L,theta_size=NULL,
                      basis_type="generic"){
    self$input_size <- input_size; self$horizon <- horizon; self$basis_type <- basis_type
    if(is.null(theta_size))
      theta_size <- if(basis_type=="generic") 2L*horizon
                    else max(3L,as.integer(log2(horizon))+1L)
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
      residual <- residual-out$backcast
      forecast <- forecast+out$forecast
    }
    forecast$unsqueeze(2L)
  }
)


# ================================================================
# ██  PART 6 — TRAINING LOOP (shared for RNN + N-BEATS)
# ================================================================

train_deep_model <- function(X_train,y_train,X_val=NULL,y_val=NULL,
                              lags,arch="lstm",
                              hidden_size=64L,num_layers=2L,dropout=0.1,
                              dense_units=32L,rnn_nonlinearity="tanh",
                              nbeats_stacks=2L,nbeats_blocks=3L,
                              nbeats_width=512L,nbeats_layers=4L,
                              nbeats_theta=NULL,nbeats_basis="generic",
                              lr=1e-3,epochs=150L,batch_size=32L,
                              patience=20L,lr_factor=0.5,lr_patience=NULL,
                              grad_clip=1.0,horizon=1L,
                              verbose=0L,device=get_device()){
  t0 <- .tic()
  if(is.null(lr_patience)) lr_patience <- max(5L,patience%/%3L)
  al <- tolower(arch)
  is_nb <- al=="nbeats"

  model <- if(!is_nb)
    StackedRNN(al,1L,as.integer(hidden_size),as.integer(num_layers),dropout,
               as.integer(dense_units),rnn_nonlinearity)$to(device=device)
  else
    NBEATSModel(lags,horizon,nbeats_stacks,nbeats_blocks,nbeats_width,nbeats_layers,
                nbeats_theta,nbeats_basis)$to(device=device)

  optimizer <- optim_adam(model$parameters,lr=lr)
  scheduler <- lr_reduce_on_plateau(optimizer,mode="min",factor=lr_factor,
                                     patience=lr_patience,min_lr=1e-7,verbose=FALSE)
  loss_fn   <- nn_mse_loss()

  make_dl <- function(Xm,yv){
    if(is_nb){
      ds <- dataset(
        initialize=function(X,y){
          self$X <- torch_tensor(X,dtype=torch_float())
          self$y <- torch_tensor(as.numeric(y),dtype=torch_float())$unsqueeze(2L)
        },
        .getitem=function(i) list(x=self$X[i,],y=self$y[i,]),
        .length=function() dim(self$X)[1L]
      )(Xm,yv)
    } else ds <-TimeSeriesDataset(Xm,yv)
    dataloader(ds,batch_size=batch_size,shuffle=FALSE)
  }

  tr_dl   <- make_dl(X_train,y_train)
  use_val <- !is.null(X_val)&&length(y_val)>0
  if(use_val) vl_dl <- make_dl(X_val,y_val)

  history <- data.frame(epoch=integer(),train_loss=double(),val_loss=double())
  best_vl <- Inf; best_st <- NULL; pat_ctr <- 0L

  for(ep in seq_len(epochs)){
    model$train(); tr_sum <- 0.0; n_tr <- 0L
    coro::loop(for(b in tr_dl){
      optimizer$zero_grad()
      xd   <- b$x$to(device=device)
      pred <- if(is_nb) model(xd)[,1L,] else model(xd)
      loss <- loss_fn(pred,b$y$to(device=device))
      loss$backward()
      if(!is.null(grad_clip)) nn_utils_clip_grad_norm_(model$parameters,max_norm=grad_clip)
      optimizer$step()
      tr_sum <- tr_sum+loss$item(); n_tr <- n_tr+1L
    })
    tr_loss <- tr_sum/n_tr; vl_loss <- NA_real_
    if(use_val){
      model$eval(); vl_sum <- 0.0; n_vl <- 0L
      with_no_grad({
        coro::loop(for(b in vl_dl){
          xd <- b$x$to(device=device)
          pred <- if(is_nb) model(xd)[,1L,] else model(xd)
          vl_sum <- vl_sum+loss_fn(pred,b$y$to(device=device))$item()
          n_vl <- n_vl+1L
        })
      })
      vl_loss <- vl_sum/n_vl; scheduler$step(vl_loss)
      if(vl_loss<best_vl-1e-6){
        best_vl <- vl_loss
        best_st <- lapply(model$state_dict(),function(t) t$clone())
        pat_ctr <- 0L
      } else pat_ctr <- pat_ctr+1L
    }
    history <- rbind(history,data.frame(epoch=ep,train_loss=tr_loss,val_loss=vl_loss))
    if(verbose>0L&&ep%%max(1L,epochs%/%10L)==0L)
      cat(sprintf("  [%s] ep%4d|tr=%.5f|vl=%.5f\n",toupper(al),ep,tr_loss,
                  ifelse(is.na(vl_loss),0,vl_loss)))
    if(use_val&&pat_ctr>=patience){
      if(verbose>0L) cat(sprintf("  Early stop ep%d\n",ep)); break}
  }
  if(!is.null(best_st)) model$load_state_dict(best_st)
  list(model=model,history=history,elapsed_sec=.toc(t0))
}


# ================================================================
# ██  PART 7 — INFERENCE
# ================================================================

predict_deep <- function(model,X_mat,device=get_device(),is_nbeats=FALSE){
  model$eval()
  X_t <- if(is_nbeats) torch_tensor(X_mat,dtype=torch_float())$to(device=device)
         else torch_tensor(X_mat,dtype=torch_float())$unsqueeze(3L)$to(device=device)
  with_no_grad({out <- model(X_t)})
  if(is_nbeats) as.numeric(out[,1L,]$squeeze()$cpu())
  else          as.numeric(out$squeeze(2L)$cpu())
}

recursive_forecast <- function(model,last_window,horizon,scaler,device=get_device()){
  model$eval(); window <- last_window; preds <- numeric(horizon)
  for(h in seq_len(horizon)){
    X_t <- torch_tensor(matrix(window,nrow=1L),dtype=torch_float())$
      unsqueeze(3L)$to(device=device)
    with_no_grad({p <- model(X_t)})
    preds[h] <- as.numeric(p$item())
    window   <- c(window[-1L],preds[h])
  }
  unscale_x(preds,scaler)
}

nbeats_forecast <- function(model,last_window,horizon,scaler,device=get_device()){
  model$eval()
  X_t <- torch_tensor(matrix(last_window,nrow=1L),dtype=torch_float())$to(device=device)
  with_no_grad({out <- model(X_t)})
  preds <- as.numeric(out$squeeze()$cpu())
  unscale_x(preds[seq_len(horizon)],scaler)
}

train_direct_models <- function(scaled_series,lags,horizon,hp,
                                 arch="lstm",epochs=150L,patience=20L,
                                 val_fraction=0.15,lr_factor=0.5,
                                 lr_patience=NULL,grad_clip=1.0,
                                 device=get_device(),verbose=0L,
                                 use_parallel=FALSE){
  cat(sprintf("\n── Direct [%s] h=1…%d %s\n",toupper(arch),horizon,
              if(use_parallel&&.is_parallel())"[parallel]" else ""))
  one_h <- function(h_step){
    sv <- make_supervised_direct(scaled_series,lags,h_step)
    n  <- nrow(sv$X); nv <- floor(n*val_fraction); nt <- n-nv
    r  <- train_deep_model(sv$X[1:nt,,drop=FALSE],sv$y[1:nt],
             if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
             if(nv>0) sv$y[(nt+1):n] else NULL,
             lags=lags,arch=arch,hidden_size=hp$hidden_size,
             num_layers=hp$num_layers,dropout=hp$dropout,dense_units=hp$dense_units,
             lr=hp$lr,epochs=epochs,batch_size=hp$batch_size,patience=patience,
             lr_factor=lr_factor,lr_patience=lr_patience,grad_clip=grad_clip,
             horizon=1L,verbose=0L,device=device)
    r$model
  }
  if(use_parallel&&.is_parallel())
    furrr::future_map(seq_len(horizon),function(h){
      suppressPackageStartupMessages(library(torch)); one_h(h)},
      .options=furrr::furrr_options(seed=TRUE,globals=FALSE),.progress=TRUE)
  else {
    models <- lapply(seq_len(horizon),function(h){cat(sprintf("  h=%2d/%d\r",h,horizon));one_h(h)})
    cat("\n"); models
  }
}

direct_forecast <- function(direct_models,last_window,scaler,device=get_device()){
  X  <- matrix(last_window,nrow=1L)
  ps <- sapply(direct_models,function(m) predict_deep(m,X,device)[1L])
  unscale_x(ps,scaler)
}


# ================================================================
# ██  PART 8 — BAYESIAN HP OPTIMISATION
# ================================================================

.bo_optimize <- function(scaled_series,lags,arch,val_fraction,
                          n_iter,init_points,epochs,patience,
                          acq,kappa,bounds,device,seed,
                          horizon=1L,use_parallel=FALSE){
  t0 <- .tic(); set.seed(seed); arch_l <- tolower(arch)
  cat(sprintf("\n── BO [%s] | acq=%s | init=%d | iter=%d%s\n",
    toupper(arch),acq,init_points,n_iter,
    if(use_parallel&&.is_parallel()) sprintf(" | workers=%d",.n_workers()) else ""))

  sv <- make_supervised(scaled_series,lags)
  n  <- nrow(sv$X); nv <- floor(n*val_fraction); nt <- n-nv
  Xt <- sv$X[1:nt,,drop=FALSE]; yt <- sv$y[1:nt]
  Xv <- sv$X[(nt+1):n,,drop=FALSE]; yv <- sv$y[(nt+1):n]

 eval_hp <- function(hp_list) {
    res <- tryCatch({
      r <- train_deep_model(
        Xt, yt, Xv, yv, lags = lags, arch = arch_l,
        hidden_size = hp_list$hidden_size, num_layers = hp_list$num_layers,
        dropout = hp_list$dropout, dense_units = hp_list$dense_units, 
        lr = hp_list$lr, epochs = epochs, batch_size = hp_list$batch_size, 
        patience = patience, horizon = horizon, verbose = 0L, device = device
      )
      
      # Extract the best validation loss
      val_loss <- min(r$history$val_loss, na.rm = TRUE)
      
      # If the model didn't converge or returned NaN/Inf, penalize it
      if (!is.finite(val_loss)) -9999 else -val_loss
      
    }, error = function(e) {
      # Log the error if needed and return a penalty score
      if (verbose > 0) message("BO Step Failed: ", e$message)
      -9999
    })
    return(res)
  }

  set.seed(seed)
  init_configs <- lapply(seq_len(init_points),function(i) list(
    hidden_size=as.integer(round(runif(1,bounds$hidden_size[1],bounds$hidden_size[2]))),
    num_layers=as.integer(round(runif(1,bounds$num_layers[1],bounds$num_layers[2]))),
    dropout=runif(1,bounds$dropout[1],bounds$dropout[2]),
    dense_units=as.integer(round(runif(1,bounds$dense_units[1],bounds$dense_units[2]))),
    lr=10^runif(1,bounds$lr_log10[1],bounds$lr_log10[2]),
    batch_size=as.integer(2^round(runif(1,bounds$batch_size_log2[1],bounds$batch_size_log2[2])))))

  init_scores <- if(use_parallel&&.is_parallel())
    furrr::future_map_dbl(init_configs,
      function(hp){suppressPackageStartupMessages(library(torch));eval_hp(hp)},
      .options=furrr::furrr_options(seed=seed,globals=FALSE),.progress=TRUE)
  else vapply(init_configs,eval_hp,numeric(1L))

  obj_seq <- function(hidden_size,num_layers,dropout,dense_units,lr_log10,batch_size_log2)
    list(Score=eval_hp(list(
      hidden_size=as.integer(round(hidden_size)),num_layers=as.integer(round(num_layers)),
      dropout=dropout,dense_units=as.integer(round(dense_units)),
      lr=10^lr_log10,batch_size=as.integer(2^round(batch_size_log2)))),Pred=0)

  init_df <- do.call(rbind,lapply(seq_along(init_configs),function(i){
    hp <- init_configs[[i]]
    data.frame(hidden_size=hp$hidden_size,num_layers=hp$num_layers,dropout=hp$dropout,
               dense_units=hp$dense_units,lr_log10=log10(hp$lr),
               batch_size_log2=log2(hp$batch_size),Value=init_scores[i])
  }))
  bo <- tryCatch(
    BayesianOptimization(FUN=obj_seq,bounds=bounds,init_points=0L,n_iter=n_iter,
                          init_grid_dt=init_df,acq=acq,kappa=kappa,verbose=TRUE),
    error=function(e) BayesianOptimization(FUN=obj_seq,bounds=bounds,
      init_points=max(3L,init_points%/%2L),n_iter=n_iter,acq=acq,kappa=kappa,verbose=TRUE))

  ws_best_idx <- which.max(init_scores)
  if(!is.infinite(init_scores[ws_best_idx])&&init_scores[ws_best_idx]>bo$Best_Value){
    hp0   <- init_configs[[ws_best_idx]]
    best  <- list(hidden_size=hp0$hidden_size,num_layers=hp0$num_layers,
                  dropout=hp0$dropout,dense_units=hp0$dense_units,
                  lr=hp0$lr,batch_size=hp0$batch_size)
  } else {
    bp   <- bo$Best_Par
    best <- list(hidden_size=as.integer(round(bp["hidden_size"])),
                 num_layers=as.integer(round(bp["num_layers"])),
                 dropout=bp["dropout"],dense_units=as.integer(round(bp["dense_units"])),
                 lr=10^bp["lr_log10"],batch_size=as.integer(2^round(bp["batch_size_log2"])))
  }
  cat(sprintf("\n── [%s] Best HP ──\n",toupper(arch))); str(best)
  list(best_params=best,bo_result=bo,elapsed_sec=.toc(t0))
}


# ================================================================
# ██  PART 9 — tsCV ENGINE  (replaces rolling_origin)
# ================================================================
#
# forecast::tsCV(y, forecastfunction, h, initial, window, ...)
# returns an error MATRIX of shape [length(y), h]:
#   rows = time index (same as y)
#   cols = forecast horizon 1…h
#   entry (t, s) = y[t+s-1] - yhat_{t-1}(s)   (h-step error)
#   NAs for positions with insufficient training history
#
# This replaces all manual fold loops and gives identical
# rolling-origin OOS evaluation with a canonical, well-tested
# implementation.
#
# Wrappers below convert forecast objects for deep learning
# models so they are compatible with the tsCV interface.
# ================================================================

#' Compute per-horizon metrics from a tsCV error matrix
#' @param e_mat  Matrix [T × h] of forecast errors from tsCV
#' @return Tibble: h, RMSE, MAE, MAPE, SMAPE, MBE, n
.tscv_metrics <- function(e_mat, y_actual) {
  h_max <- ncol(e_mat)
  rows  <- lapply(seq_len(h_max), function(h_step) {
    e   <- e_mat[, h_step]
    idx <- which(!is.na(e))
    if (length(idx) < 5L)
      return(tibble(h=h_step, RMSE=NA_real_, MAE=NA_real_,
                    MAPE=NA_real_, SMAPE=NA_real_, MBE=NA_real_, n=0L))
    e_h <- e[idx]
    # actual at position t+h-1 (offset from error matrix convention)
    act_idx <- idx + h_step - 1L
    act_idx <- act_idx[act_idx <= length(y_actual)]
    e_h     <- e_h[seq_along(act_idx)]
    a_h     <- y_actual[act_idx]
    p_h     <- a_h - e_h
    tibble(
      h     = h_step,
      RMSE  = sqrt(mean(e_h^2, na.rm=TRUE)),
      MAE   = mean(abs(e_h),   na.rm=TRUE),
      MAPE  = if(any(a_h==0,na.rm=TRUE)) NA_real_
              else mean(abs(e_h/a_h)*100, na.rm=TRUE),
      SMAPE = mean(200*abs(e_h)/(abs(a_h)+abs(p_h)), na.rm=TRUE),
      MBE   = mean(e_h, na.rm=TRUE),
      n     = length(e_h)
    )
  })
  dplyr::bind_rows(rows)
}

#' Build tidy OOS tibble from tsCV error matrix
.tscv_oos_tibble <- function(e_mat, ts_y, dates_vec, strategy_name) {
  n   <- nrow(e_mat)
  h_max <- ncol(e_mat)
  rows  <- lapply(seq_len(h_max), function(h_step) {
    e_col <- e_mat[, h_step]
    idx   <- which(!is.na(e_col))
    if(length(idx) == 0L) return(NULL)
    act_idx <- pmin(idx + h_step - 1L, n)
    a_v  <- as.numeric(ts_y)[act_idx]
    e_v  <- e_col[idx]
    p_v  <- a_v - e_v
    tibble(
      fold      = idx,
      h         = h_step,
      date      = dates_vec[act_idx],
      actual    = a_v,
      predicted = p_v,
      residual  = e_v,
      strategy  = strategy_name
    )
  })
  dplyr::bind_rows(Filter(Negate(is.null), rows))
}

#' Build aggregate CV summary from tsCV metrics tibble
.tscv_summary <- function(metrics_tbl, strategy_name) {
  metrics_tbl %>%
    summarise(
      across(c(RMSE, MAE, MAPE, SMAPE, MBE),
             list(mean = ~mean(.x, na.rm=TRUE),
                  sd   = ~sd(.x,   na.rm=TRUE)),
             .names = paste0("{.col}_", strategy_name, "__{.fn}"))
    ) %>%
    pivot_longer(everything(), names_to=c("metric","stat"), names_sep="__") %>%
    pivot_wider(names_from=stat, values_from=value)
}


#' Rolling-origin CV via forecast::tsCV for deep learning models
#'
#' Creates a forecastfunction wrapper around the deep learning
#' model and passes it to forecast::tsCV(). For each origin t,
#' tsCV trains a new model on y[1:t] and predicts h steps ahead.
#'
#' @param data         Data frame (date, value).
#' @param lags         Input window size.
#' @param horizon      Forecast horizon h.
#' @param hp           Hyperparameter list.
#' @param arch         Architecture string.
#' @param cv_initial   Min training observations (tsCV `initial` arg).
#' @param cv_window    Fixed window size (NULL = expanding).
#' @param strategy     "recursive" | "direct" | "both".
#' @param epochs,patience,val_fraction,lr_factor,lr_patience,
#'        grad_clip,scale,device,verbose  Training settings.
#' @param use_parallel Use furrr parallel (if .is_parallel()).
#' @return List: cv_metrics, by_horizon, summary, oos_tbl, elapsed_sec
.tscv_deep <- function(data, lags, horizon, hp, arch,
                        cv_initial, cv_window, strategy,
                        epochs, patience, val_fraction, lr_factor,
                        lr_patience, grad_clip, scale, device, verbose,
                        use_parallel=FALSE) {
  t0     <- .tic()
  arch_l <- tolower(arch)
  ts_y   <- .df_to_ts(data)
  n_obs  <- length(ts_y)
  dates  <- data$date
  cat(sprintf("\n── tsCV [%s] | h=%d | initial=%s | window=%s | strategy=%s\n",
    toupper(arch_l), horizon,
    cv_initial %||% "auto",
    ifelse(is.null(cv_window), "expanding", cv_window),
    strategy))

  # Auto cv_initial
  if (is.null(cv_initial))
    cv_initial <- max(lags + horizon + 5L, as.integer(n_obs * 0.5))

  oos_list   <- list()
  e_mat_list <- list()

  for (strat in c("recursive","direct")) {
    if (!strat %in% c(strategy, "both") && strategy != "both") next
    if (strat == "direct" && arch_l == "nbeats") next  # nbeats always uses its own direct

    cat(sprintf("  tsCV [%s - %s]…\n", toupper(arch_l), strat))

    # ── Build forecastfunction for tsCV ──────────────────────────
    # tsCV calls forecastfunction(y_train, h) at each origin.
    # We train a fresh model on y_train and forecast h steps.
    fc_fun <- local({
      # Capture all needed args in local env
      lags_   <- lags; arch_l_ <- arch_l; hp_     <- hp
      ep_     <- epochs; pat_   <- patience; vf_   <- val_fraction
      lrf_    <- lr_factor; lrp_  <- lr_patience; gc_ <- grad_clip
      sc_     <- scale; dev_  <- device; s_ <- strat

      function(y_train, h, ...) {
        suppressPackageStartupMessages(library(torch))
        n_tr   <- length(y_train)
        sc_par <- list(mean=mean(y_train,na.rm=TRUE), sd=sd(y_train,na.rm=TRUE))
        y_sc   <- if(sc_) (y_train-sc_par$mean)/sc_par$sd else y_train
        last_w <- tail(y_sc, lags_)

        if(s_=="recursive") {
          # Train 1-step model
          sv <- make_supervised(y_sc, lags_)
          n  <- nrow(sv$X); nv <- floor(n*vf_); nt <- n-nv
          if(nt < 5L) return(forecast::forecast(forecast::rwf(y_train), h=h))
          res <- train_deep_model(
            sv$X[1:nt,,drop=FALSE], sv$y[1:nt],
            if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
            if(nv>0) sv$y[(nt+1):n] else NULL,
            lags=lags_, arch=arch_l_, hidden_size=hp_$hidden_size,
            num_layers=hp_$num_layers, dropout=hp_$dropout, dense_units=hp_$dense_units,
            lr=hp_$lr, epochs=ep_, batch_size=hp_$batch_size, patience=pat_,
            lr_factor=lrf_, lr_patience=lrp_, grad_clip=gc_, horizon=1L,
            verbose=0L, device=dev_)
          fc_vals <- recursive_forecast(res$model, last_w, h, sc_par, dev_)
        } else {
          # Direct: train H separate 1-step models targeting each horizon
          fc_vals <- numeric(h)
          for(h_step in seq_len(h)){
            sv_h <- make_supervised_direct(y_sc, lags_, h_step)
            n_h  <- nrow(sv_h$X)
            if(n_h < 5L){ fc_vals[h_step] <- NA_real_; next }
            nv_h <- floor(n_h*vf_); nt_h <- n_h-nv_h
            res_h <- train_deep_model(
              sv_h$X[1:nt_h,,drop=FALSE], sv_h$y[1:nt_h],
              if(nv_h>0) sv_h$X[(nt_h+1):n_h,,drop=FALSE] else NULL,
              if(nv_h>0) sv_h$y[(nt_h+1):n_h] else NULL,
              lags=lags_, arch=arch_l_, hidden_size=hp_$hidden_size,
              num_layers=hp_$num_layers, dropout=hp_$dropout, dense_units=hp_$dense_units,
              lr=hp_$lr, epochs=ep_, batch_size=hp_$batch_size, patience=pat_,
              lr_factor=lrf_, lr_patience=lrp_, grad_clip=gc_, horizon=1L,
              verbose=0L, device=dev_)
            X_pred <- matrix(last_w, nrow=1L)
            fc_vals[h_step] <- (sc_par$sd * predict_deep(res_h$model, X_pred, dev_)[1L]) +
                                sc_par$mean
          }
        }
        # Return as forecast object compatible with tsCV
        structure(
          list(mean = ts(fc_vals, start = n_tr + 1L, frequency = frequency(y_train))),
          class = "forecast"
        )
      }
    })

    # ── Call forecast::tsCV ──────────────────────────────────────
    e_mat <- tryCatch(
      forecast::tsCV(y = ts_y, forecastfunction = fc_fun,
                     h = horizon, initial = cv_initial,
                     window = cv_window),
      error = function(e) {
        warning(sprintf("tsCV [%s %s] failed: %s", arch_l, strat, e$message))
        NULL
      }
    )
    if (is.null(e_mat)) next

    e_mat_list[[strat]] <- e_mat
    oos_list[[strat]]   <- .tscv_oos_tibble(e_mat, ts_y, dates, strat)
  }

  # N-BEATS direct H-step via tsCV
  if (arch_l == "nbeats" && strategy %in% c("recursive","both","direct")) {
    cat(sprintf("  tsCV [N-BEATS direct H-step]…\n"))
    fc_fun_nb <- local({
      lags_ <- lags; ep_ <- epochs; pat_ <- patience; vf_ <- val_fraction
      lrf_  <- lr_factor; lrp_ <- lr_patience; gc_ <- grad_clip
      sc_   <- scale; dev_ <- device; hp_ <- hp

      function(y_train, h, ...) {
        suppressPackageStartupMessages(library(torch))
        sc_par <- list(mean=mean(y_train,na.rm=TRUE), sd=sd(y_train,na.rm=TRUE))
        y_sc   <- if(sc_) (y_train-sc_par$mean)/sc_par$sd else y_train
        sv <- make_supervised(y_sc, lags_)
        n  <- nrow(sv$X); nv <- floor(n*vf_); nt <- n-nv
        if(nt < 5L) return(forecast::forecast(forecast::rwf(y_train), h=h))
        res <- train_deep_model(
          sv$X[1:nt,,drop=FALSE], sv$y[1:nt],
          if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
          if(nv>0) sv$y[(nt+1):n] else NULL,
          lags=lags_, arch="nbeats", hidden_size=hp_$hidden_size,
          num_layers=hp_$num_layers, dropout=hp_$dropout, dense_units=hp_$dense_units,
          lr=hp_$lr, epochs=ep_, batch_size=hp_$batch_size, patience=pat_,
          lr_factor=lrf_, lr_patience=lrp_, grad_clip=gc_, horizon=h,
          verbose=0L, device=dev_)
        last_w  <- tail(y_sc, lags_)
        fc_vals <- nbeats_forecast(res$model, last_w, h, sc_par, dev_)
        n_tr    <- length(y_train)
        structure(
          list(mean = ts(fc_vals, start=n_tr+1L, frequency=frequency(y_train))),
          class = "forecast"
        )
      }
    })
    e_mat_nb <- tryCatch(
      forecast::tsCV(y=ts_y, forecastfunction=fc_fun_nb,
                     h=horizon, initial=cv_initial, window=cv_window),
      error=function(e){ warning(sprintf("tsCV N-BEATS failed: %s",e$message)); NULL})
    if (!is.null(e_mat_nb)) {
      e_mat_list[["nbeats"]] <- e_mat_nb
      oos_list[["nbeats"]]   <- .tscv_oos_tibble(e_mat_nb, ts_y, dates, "nbeats")
    }
  }

  if (length(e_mat_list) == 0L)
    stop("tsCV produced no results for any strategy.")

  # ── Aggregate per-horizon and overall metrics ─────────────────
  by_horizon_list <- lapply(names(e_mat_list), function(s)
    .tscv_metrics(e_mat_list[[s]], as.numeric(ts_y)) %>% mutate(strategy = s))
  by_horizon_df <- dplyr::bind_rows(by_horizon_list)

  overall_list <- lapply(names(e_mat_list), function(s) {
    bh <- by_horizon_df[by_horizon_df$strategy == s, ]
    tibble(
      strategy = s,
      RMSE     = mean(bh$RMSE, na.rm=TRUE),
      MAE      = mean(bh$MAE,  na.rm=TRUE),
      MAPE     = mean(bh$MAPE, na.rm=TRUE),
      SMAPE    = mean(bh$SMAPE,na.rm=TRUE),
      MBE      = mean(bh$MBE,  na.rm=TRUE),
      n_horizons = sum(!is.na(bh$RMSE))
    )
  })
  cv_metrics  <- dplyr::bind_rows(overall_list)
  oos_combined <- dplyr::bind_rows(oos_list)

  cat(sprintf("\n── [%s] tsCV Summary ──\n", toupper(arch_l)))
  print(cv_metrics, n=Inf)

  list(
    cv_metrics   = cv_metrics,
    by_horizon   = by_horizon_df,
    oos_tbl      = oos_combined,
    e_matrices   = e_mat_list,
    elapsed_sec  = .toc(t0)
  )
}

# ================================================================
# ██  PART 10 — HOLD-OUT OOS EVALUATION
# ================================================================

.holdout_eval <- function(data,lags,horizon,hp,arch,
                           holdout_frac,strategy,epochs,patience,
                           val_fraction,lr_factor,lr_patience,grad_clip,
                           scale,device,verbose){
  t0 <- .tic(); arch_l <- tolower(arch)
  n_total <- nrow(data); n_ho <- max(horizon,floor(n_total*holdout_frac))
  n_train <- n_total-n_ho
  if(n_train<lags+horizon+5L) stop("Hold-out too large. Reduce cv_holdout_frac.")
  tr_df  <- data[seq_len(n_train),]; tst_df <- data[(n_train+1L):n_total,]
  cat(sprintf("\n── Hold-out [%s] | train=%d | test=%d\n",toupper(arch_l),n_train,nrow(tst_df)))
  sc     <- make_scaler(tr_df$value)
  tr_sc  <- if(scale) scale_x(tr_df$value,sc) else tr_df$value
  tst_sc <- if(scale) scale_x(tst_df$value,sc) else tst_df$value
  all_sc <- c(tr_sc,tst_sc)
  sv     <- make_supervised(tr_sc,lags)
  n      <- nrow(sv$X); nv <- floor(n*val_fraction); nt <- n-nv
  results <- list()

  .train_tr <- function() train_deep_model(
    sv$X[1:nt,,drop=FALSE],sv$y[1:nt],
    if(nv>0) sv$X[(nt+1):n,,drop=FALSE] else NULL,
    if(nv>0) sv$y[(nt+1):n] else NULL,
    lags=lags,arch=arch_l,hidden_size=hp$hidden_size,num_layers=hp$num_layers,
    dropout=hp$dropout,dense_units=hp$dense_units,lr=hp$lr,
    epochs=epochs,batch_size=hp$batch_size,patience=patience,
    lr_factor=lr_factor,lr_patience=lr_patience,grad_clip=grad_clip,
    horizon=1L,verbose=verbose,device=device)

  # Rolling 1-step-ahead true OOS fitted values
  .rolling_fit <- function(mdl){
    n_tst <- length(tst_sc); preds <- numeric(n_tst)
    for(i in seq_len(n_tst)){
      ws <- n_train+i-1L; wst <- ws-lags+1L
      if(wst<1L){preds[i]<-NA_real_;next}
      win <- all_sc[wst:ws]
      X_t <- torch_tensor(matrix(win,nrow=1L),dtype=torch_float())$
        unsqueeze(3L)$to(device=device)
      mdl$eval(); with_no_grad({p<-mdl(X_t)}); preds[i]<-as.numeric(p$item())
    }
    unscale_x(preds,sc)
  }

  if(strategy%in%c("recursive","both")){
    res <- .train_tr(); fv <- .rolling_fit(res$model)
    m   <- .safe_metrics(tst_df$value,fv)
    cat(sprintf("  [Recursive] RMSE=%.4f|MAE=%.4f|MAPE=%.2f%%\n",m$RMSE,m$MAE,m$MAPE%||%NA))
    results$recursive <- tibble(fold=1L,h=seq_len(nrow(tst_df)),date=tst_df$date,
      actual=tst_df$value,fitted=fv,residual=tst_df$value-fv,strategy="recursive")
    results$metrics_recursive <- m
    results$model_recursive   <- res$model
    results$history_recursive <- res$history
  }
  if(strategy%in%c("direct","both")){
    sv1 <- make_supervised_direct(tr_sc,lags,1L)
    n1  <- nrow(sv1$X); nv1 <- floor(n1*val_fraction); nt1 <- n1-nv1
    r1  <- train_deep_model(sv1$X[1:nt1,,drop=FALSE],sv1$y[1:nt1],
             if(nv1>0) sv1$X[(nt1+1):n1,,drop=FALSE] else NULL,
             if(nv1>0) sv1$y[(nt1+1):n1] else NULL,
             lags=lags,arch=arch_l,hidden_size=hp$hidden_size,num_layers=hp$num_layers,
             dropout=hp$dropout,dense_units=hp$dense_units,lr=hp$lr,
             epochs=epochs,batch_size=hp$batch_size,patience=patience,
             lr_factor=lr_factor,lr_patience=lr_patience,grad_clip=grad_clip,
             horizon=1L,verbose=0L,device=device)
    fv_d <- .rolling_fit(r1$model); m_d <- .safe_metrics(tst_df$value,fv_d)
    cat(sprintf("  [Direct]    RMSE=%.4f|MAE=%.4f|MAPE=%.2f%%\n",m_d$RMSE,m_d$MAE,m_d$MAPE%||%NA))
    results$direct <- tibble(fold=1L,h=seq_len(nrow(tst_df)),date=tst_df$date,
      actual=tst_df$value,fitted=fv_d,residual=tst_df$value-fv_d,strategy="direct")
    results$metrics_direct <- m_d
  }
  results$elapsed_sec <- .toc(t0)
  results$n_train <- n_train; results$n_test <- nrow(tst_df)
  results$train_df <- tr_df; results$test_df <- tst_df
  results
}


# ================================================================
# ██  PART 11 — MASTER: auto_deep() / auto_lstm / auto_gru /
#                       auto_rnn / auto_nbeats
# ================================================================

#' @export
auto_deep <- function(data,
                       date_col="date",value_col="value",
                       arch="lstm",horizon=12L,
                       strategy=c("both","recursive","direct"),
                       device=NULL,config=NULL,
                       run_bo=NULL,bo_init=NULL,bo_iter=NULL,
                       bo_epochs=NULL,bo_patience=NULL,bo_bounds=NULL,
                       final_epochs=NULL,final_patience=NULL,
                       run_cv=NULL,cv_initial=NULL,cv_window=NULL,
                       cv_holdout_frac=NULL,
                       val_fraction=NULL,lags=NULL,scale=NULL,
                       seed=NULL,verbose=NULL,use_parallel=NULL){
  total_t0 <- .tic()
  cfg <- .resolve_cfg(config,list(
    run_bo=run_bo,bo_init=bo_init,bo_iter=bo_iter,
    bo_epochs=bo_epochs,bo_patience=bo_patience,bo_bounds=bo_bounds,
    final_epochs=final_epochs,final_patience=final_patience,
    run_cv=run_cv,cv_initial=cv_initial,cv_window=cv_window,
    cv_holdout_frac=cv_holdout_frac,val_fraction=val_fraction,
    lags=lags,scale=scale,seed=seed,verbose=verbose))
  set.seed(cfg$seed); torch_manual_seed(cfg$seed)
  arch_l   <- .check_arch(arch)
  strategy <- match.arg(strategy)
  if(is.null(device)) device <- get_device()
  use_par  <- (use_parallel %||% TRUE) && .is_parallel() && device=="cpu"
  if(is.null(cfg$lags)) cfg$lags <- max(12L,as.integer(ceiling(horizon*1.5)))
  lags_v <- cfg$lags

  stopifnot(is.data.frame(data),date_col%in%names(data),value_col%in%names(data))
  df <- data%>%rename(date=!!sym(date_col),value=!!sym(value_col))%>%
    arrange(date)%>%select(date,value)
  if(anyNA(df$value)){warning("Missing→interpolating.");df$value<-zoo::na.approx(df$value,na.rm=FALSE)}

  cat(sprintf("\nAuto%s|n=%d|h=%d|lags=%d|strategy=%s|device=%s|par=%s\n",
    toupper(arch_l),nrow(df),horizon,lags_v,strategy,device,use_par))
  cat(sprintf("  BO=%s | CV=%s%s\n",
    if(cfg$run_bo)"ON" else "OFF",
    if(cfg$run_cv)"tsCV" else sprintf("hold-out %.0f%%",cfg$cv_holdout_frac*100),
    if(cfg$run_cv&&!is.null(cfg$cv_window)) sprintf("[window=%d]",cfg$cv_window) else
    if(cfg$run_cv) "[expanding]" else ""))

  sc      <- make_scaler(df$value)
  scaled  <- if(cfg$scale) scale_x(df$value,sc) else df$value
  def_hp  <- list(hidden_size=cfg$hidden_size,num_layers=cfg$num_layers,
                  dropout=cfg$dropout,dense_units=cfg$dense_units,
                  lr=cfg$lr,batch_size=cfg$batch_size)
  runtime <- list()

  # ── Bayesian Optimisation ─────────────────────────────────────
  bo_result <- NULL
  if(cfg$run_bo){
    bounds  <- .resolve_bounds(cfg$bo_bounds)
    bo_out  <- .bo_optimize(scaled,lags_v,arch_l,cfg$val_fraction,
                             cfg$bo_iter,cfg$bo_init,cfg$bo_epochs,cfg$bo_patience,
                             cfg$bo_acq,cfg$bo_kappa,bounds,device,cfg$seed,
                             horizon=horizon,use_parallel=use_par)
    best_hp <- bo_out$best_params; bo_result <- bo_out$bo_result
    runtime$bo_sec <- bo_out$elapsed_sec
  } else { best_hp <- def_hp; runtime$bo_sec <- 0 }

  # ── tsCV or hold-out ─────────────────────────────────────────
  cv_results <- NULL; holdout_results <- NULL

  if(cfg$run_cv){
    cv_out <- .tscv_deep(df,lags_v,horizon,best_hp,arch_l,
                          cfg$cv_initial,cfg$cv_window,strategy,
                          cfg$final_epochs,cfg$final_patience,
                          cfg$val_fraction,cfg$lr_factor,cfg$lr_patience,
                          cfg$grad_clip,cfg$scale,device,0L,use_parallel=use_par)
    cv_results   <- cv_out; runtime$cv_sec <- cv_out$elapsed_sec
  } else {
    ho_out <- .holdout_eval(df,lags_v,horizon,best_hp,arch_l,
                             cfg$cv_holdout_frac,strategy,
                             cfg$final_epochs,cfg$final_patience,
                             cfg$val_fraction,cfg$lr_factor,cfg$lr_patience,
                             cfg$grad_clip,cfg$scale,device,cfg$verbose)
    holdout_results <- ho_out; runtime$holdout_sec <- ho_out$elapsed_sec
  }

  # ── Final model on full series ────────────────────────────────
  cat(sprintf("\n── Final [%s] on full series ──\n",toupper(arch_l)))
  ft0     <- .tic()
  sv_full <- make_supervised(scaled,lags_v)
  n       <- nrow(sv_full$X); nv <- floor(n*cfg$val_fraction); nt <- n-nv
  X_tr    <- sv_full$X[1:nt,,drop=FALSE]; y_tr <- sv_full$y[1:nt]
  X_vl    <- if(nv>0) sv_full$X[(nt+1):n,,drop=FALSE] else NULL
  y_vl    <- if(nv>0) sv_full$y[(nt+1):n] else NULL
  last_win<- tail(scaled,lags_v)
  freq    <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date),by=freq,length.out=horizon+1L)[-1L]

  .ci <- function(mdl,Xm,yv,fc){
    is_nb <- arch_l=="nbeats"
    p     <- predict_deep(mdl,Xm,device,is_nb)
    res_sd <- sd(unscale_x(yv,sc)-unscale_x(p,sc))
    ci_w   <- qnorm(0.975)*res_sd*sqrt(seq_len(horizon))
    list(lower_95=fc-ci_w,upper_95=fc+ci_w)
  }

  model_recursive   <- NULL; history_recursive <- NULL
  forecast_rec_df   <- NULL; direct_models     <- NULL; forecast_dir_df <- NULL

  if(arch_l=="nbeats"){
    cat("  [N-BEATS] Training H-step model…\n")
    res_nb <- train_deep_model(X_tr,y_tr,X_vl,y_vl,lags=lags_v,arch="nbeats",
      hidden_size=best_hp$hidden_size,num_layers=best_hp$num_layers,
      dropout=best_hp$dropout,dense_units=best_hp$dense_units,
      lr=best_hp$lr,epochs=cfg$final_epochs,batch_size=best_hp$batch_size,
      patience=cfg$final_patience,lr_factor=cfg$lr_factor,grad_clip=cfg$grad_clip,
      horizon=horizon,verbose=cfg$verbose,device=device)
    model_recursive <- res_nb$model; history_recursive <- res_nb$history
    fc_nb <- nbeats_forecast(model_recursive,last_win,horizon,sc,device)
    ci_nb <- .ci(model_recursive,X_tr,y_tr,fc_nb)
    forecast_rec_df <- tibble(date=fut_dates,forecast=fc_nb,
      lower_95=ci_nb$lower_95,upper_95=ci_nb$upper_95,strategy="nbeats")
    cat("  [N-BEATS] Done.\n")
  } else {
    if(strategy%in%c("recursive","both")){
      cat("  [Recursive] Training…\n")
      res <- train_deep_model(X_tr,y_tr,X_vl,y_vl,lags=lags_v,arch=arch_l,
        hidden_size=best_hp$hidden_size,num_layers=best_hp$num_layers,
        dropout=best_hp$dropout,dense_units=best_hp$dense_units,lr=best_hp$lr,
        epochs=cfg$final_epochs,batch_size=best_hp$batch_size,patience=cfg$final_patience,
        lr_factor=cfg$lr_factor,lr_patience=cfg$lr_patience,grad_clip=cfg$grad_clip,
        horizon=1L,verbose=cfg$verbose,device=device)
      model_recursive <- res$model; history_recursive <- res$history
      fc <- recursive_forecast(model_recursive,last_win,horizon,sc,device)
      ci <- .ci(model_recursive,X_tr,y_tr,fc)
      forecast_rec_df <- tibble(date=fut_dates,forecast=fc,
        lower_95=ci$lower_95,upper_95=ci$upper_95,strategy="recursive")
      cat("  [Recursive] Done.\n")
    }
    if(strategy%in%c("direct","both")){
      cat("  [Direct] Training H models…\n")
      direct_models <- train_direct_models(scaled,lags_v,horizon,best_hp,arch_l,
        cfg$final_epochs,cfg$final_patience,cfg$val_fraction,
        cfg$lr_factor,cfg$lr_patience,cfg$grad_clip,device,0L,use_parallel=use_par)
      fc <- direct_forecast(direct_models,last_win,sc,device)
      sv1 <- make_supervised_direct(scaled,lags_v,1L)
      nt1 <- nrow(sv1$X)-floor(nrow(sv1$X)*cfg$val_fraction)
      p1  <- predict_deep(direct_models[[1]],sv1$X[1:nt1,,drop=FALSE],device)
      res_sd1 <- sd(unscale_x(sv1$y[1:nt1],sc)-unscale_x(p1,sc))
      ci_w1   <- qnorm(0.975)*res_sd1*sqrt(seq_len(horizon))
      forecast_dir_df <- tibble(date=fut_dates,forecast=fc,
        lower_95=fc-ci_w1,upper_95=fc+ci_w1,strategy="direct")
      cat("  [Direct] Done.\n")
    }
  }
  runtime$final_train_sec <- .toc(ft0); runtime$total_sec <- .toc(total_t0)
  cat(sprintf("\n✓ Auto%s|total=%s|BO=%s|eval=%s|final=%s\n",
    toupper(arch_l),.fmt_elapsed(runtime$total_sec),
    .fmt_elapsed(runtime$bo_sec%||%0),
    .fmt_elapsed((runtime$cv_sec%||%runtime$holdout_sec)%||%0),
    .fmt_elapsed(runtime$final_train_sec)))

  structure(list(
    model_type=arch_l,strategy=strategy,horizon=horizon,lags=lags_v,
    forecast_recursive=forecast_rec_df,forecast_direct=forecast_dir_df,
    best_params=best_hp,bo_result=bo_result,
    cv_results=cv_results,holdout_results=holdout_results,
    model_recursive=model_recursive,direct_models=direct_models,
    history_recursive=history_recursive,
    scaler=sc,data=df,config_used=cfg,runtime=runtime
  ),class=c(paste0("auto_",arch_l),"auto_rnn"))
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
# ██  PART 12 — AUTO-ARIMA  (with forecast::tsCV)
# ================================================================

#' @export
auto_arima_ts <- function(data,date_col="date",value_col="value",
                           horizon=12L,level=c(80,95),stepwise=TRUE,
                           approximation=NULL,lambda=NULL,
                           run_cv=TRUE,cv_initial=NULL,cv_window=NULL,
                           cv_holdout_frac=0.20,
                           xreg=NULL,xreg_future=NULL,seed=42L,verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  df <- data%>%rename(date=!!sym(date_col),value=!!sym(value_col))%>%
    arrange(date)%>%select(date,value)
  if(anyNA(df$value)){warning("Missing→interp.");df$value<-zoo::na.approx(df$value,na.rm=FALSE)}
  ts_obj <- .df_to_ts(df); freq <- .ts_freq(df$date)
  if(verbose>0L) cat("\n── Auto-ARIMA ─────────────────────────────────────────────\n")
  fit_t0 <- .tic()
  model  <- forecast::auto.arima(ts_obj,xreg=xreg,stepwise=stepwise,
    approximation=approximation,lambda=lambda,ic="aicc",trace=verbose>0L)
  fit_sec <- .toc(fit_t0)
  if(verbose>0L){
    cat(sprintf("\n  ARIMA(%d,%d,%d)",model$arma[1],model$arma[6],model$arma[2]))
    if(freq>1L) cat(sprintf("(%d,%d,%d)[%d]",model$arma[3],model$arma[7],model$arma[4],freq))
    cat(sprintf(" | AICc=%.2f\n",model$aicc))}
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date),by=freq_str,length.out=horizon+1L)[-1L]
  fc_obj    <- forecast::forecast(model,h=horizon,level=level,xreg=xreg_future)
  forecast_df <- tibble(date=fut_dates,forecast=as.numeric(fc_obj$mean),
    lower_80=as.numeric(fc_obj$lower[,1]),upper_80=as.numeric(fc_obj$upper[,1]),
    lower_95=if(length(level)>=2L) as.numeric(fc_obj$lower[,2]) else NA_real_,
    upper_95=if(length(level)>=2L) as.numeric(fc_obj$upper[,2]) else NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0

  if(run_cv){
    ev_t0 <- .tic()
    if(is.null(cv_initial)) cv_initial <- max(as.integer(nrow(df)*0.5),2L*freq+horizon)
    cat(sprintf("\n── Auto-ARIMA tsCV | h=%d | initial=%d | window=%s\n",
      horizon,cv_initial,ifelse(is.null(cv_window),"expanding",cv_window)))

    # forecastfunction for tsCV
    fc_fun_arima <- local({
      sw_ <- stepwise; ap_ <- approximation; lm_ <- lambda
      function(y_tr, h, ...) {
        m_i <- tryCatch(
          forecast::auto.arima(y_tr,stepwise=sw_,approximation=ap_,lambda=lm_,ic="aicc"),
          error=function(e) NULL)
        if(is.null(m_i)) return(forecast::forecast(forecast::rwf(y_tr),h=h))
        forecast::forecast(m_i,h=h)
      }
    })

    e_mat <- tryCatch(
      forecast::tsCV(y=ts_obj,forecastfunction=fc_fun_arima,
                     h=horizon,initial=cv_initial,window=cv_window),
      error=function(e){warning(sprintf("ARIMA tsCV failed: %s",e$message));NULL})

    if(!is.null(e_mat)){
      by_h    <- .tscv_metrics(e_mat,as.numeric(ts_obj))
      overall <- tibble(
        RMSE=mean(by_h$RMSE,na.rm=TRUE),MAE=mean(by_h$MAE,na.rm=TRUE),
        MAPE=mean(by_h$MAPE,na.rm=TRUE),SMAPE=mean(by_h$SMAPE,na.rm=TRUE),
        MBE=mean(by_h$MBE,na.rm=TRUE))
      oos_tbl <- .tscv_oos_tibble(e_mat,ts_obj,df$date,"arima")
      cv_results <- list(cv_metrics=overall,by_horizon=by_h,
                          oos_tbl=oos_tbl,e_matrix=e_mat)
      if(verbose>0L){ cat("\n── ARIMA tsCV Metrics ──\n"); print(overall) }
    }
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic()
    n_total <- nrow(df); n_ho <- max(horizon,floor(n_total*cv_holdout_frac))
    n_tr    <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    cat(sprintf("\n── ARIMA hold-out (train=%d,test=%d) ──\n",n_tr,nrow(tst_df)))
    ts_tr <- ts(tr_df$value,frequency=freq,start=c(as.integer(format(min(tr_df$date),"%Y")),
                  if(freq==12L) as.integer(format(min(tr_df$date),"%m")) else 1L))
    m_ho  <- tryCatch(forecast::auto.arima(ts_tr,stepwise=stepwise,ic="aicc"),error=function(e) NULL)
    n_tst <- nrow(tst_df); fv <- numeric(n_tst)
    if(!is.null(m_ho)){
      all_ts <- ts(df$value,frequency=freq,start=c(as.integer(format(min(df$date),"%Y")),
                     if(freq==12L) as.integer(format(min(df$date),"%m")) else 1L))
      refit  <- tryCatch(forecast::Arima(all_ts,model=m_ho),error=function(e) NULL)
      if(!is.null(refit)){af<-as.numeric(fitted(refit));fv<-af[(n_tr+1L):(n_tr+n_tst)]}
      else fv <- tryCatch(as.numeric(forecast::forecast(m_ho,h=n_tst)$mean),
                           error=function(e) rep(NA_real_,n_tst))
    } else fv <- rep(NA_real_,n_tst)
    m_ho <- .safe_metrics(tst_df$value,fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f|MAPE=%.2f%%\n",m_ho$RMSE,m_ho$MAE,m_ho$MAPE%||%NA))
    holdout_results <- tibble(fold=1L,h=seq_len(n_tst),date=tst_df$date,
      actual=tst_df$value,fitted=fv,residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec,eval_sec=eval_sec,total_sec=.toc(total_t0))
  cat(sprintf("\n✓ Auto-ARIMA|total=%s|fit=%s|eval=%s\n",
    .fmt_elapsed(runtime$total_sec),.fmt_elapsed(runtime$fit_sec),.fmt_elapsed(runtime$eval_sec)))
  structure(list(model=model,forecast=forecast_df,cv_results=cv_results,
    holdout_results=holdout_results,data=df,horizon=horizon,level=level,
    ts_obj=ts_obj,freq=freq,runtime=runtime),class="auto_arima_ts")
}


# ================================================================
# ██  PART 13 — AUTO-ARFIMA  (with forecast::tsCV)
# ================================================================

#' @export
auto_arfima_ts <- function(data,date_col="date",value_col="value",
                            horizon=12L,level=c(80,95),drange=c(0,0.5),
                            ar_max=5L,ma_max=5L,
                            run_cv=TRUE,cv_initial=NULL,cv_window=NULL,
                            cv_holdout_frac=0.20,seed=42L,verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  df <- data%>%rename(date=!!sym(date_col),value=!!sym(value_col))%>%
    arrange(date)%>%select(date,value)
  if(anyNA(df$value)){warning("Missing→interp.");df$value<-zoo::na.approx(df$value,na.rm=FALSE)}
  ts_obj <- .df_to_ts(df); freq <- .ts_freq(df$date)
  stat <- list()
  tryCatch({
    adf <- tseries::adf.test(ts_obj); kpss <- tseries::kpss.test(ts_obj)
    stat <- list(adf_pval=adf$p.value,kpss_pval=kpss$p.value)
    if(verbose>0L) cat(sprintf("\n── Stationarity: ADF p=%.4f|KPSS p=%.4f\n",adf$p.value,kpss$p.value))
  },error=function(e) NULL)
  fit_t0 <- .tic()
  fd_fit <- fracdiff::fracdiff(ts_obj,drange=drange,ar=ar_max,ma=ma_max)
  d_hat  <- fd_fit$d
  if(verbose>0L) cat(sprintf("  d=%.6f\n",d_hat))
  model  <- tryCatch(forecast::arfima(ts_obj,drange=drange,estim="mle"),
                     error=function(e){warning("arfima() fallback.");fd_fit})
  fit_sec <- .toc(fit_t0)
  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date),by=freq_str,length.out=horizon+1L)[-1L]
  fc_obj    <- tryCatch(forecast::forecast(model,h=horizon,level=level),error=function(e) NULL)
  forecast_df <- if(!is.null(fc_obj)) tibble(date=fut_dates,
    forecast=as.numeric(fc_obj$mean),lower_80=as.numeric(fc_obj$lower[,1]),
    upper_80=as.numeric(fc_obj$upper[,1]),
    lower_95=if(length(level)>=2L) as.numeric(fc_obj$lower[,2]) else NA_real_,
    upper_95=if(length(level)>=2L) as.numeric(fc_obj$upper[,2]) else NA_real_)
  else tibble(date=fut_dates,forecast=rep(NA_real_,horizon),
    lower_80=NA_real_,upper_80=NA_real_,lower_95=NA_real_,upper_95=NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0

  if(run_cv){
    ev_t0 <- .tic()
    if(is.null(cv_initial)) cv_initial <- max(as.integer(nrow(df)*0.5),2L*freq+horizon)
    cat(sprintf("\n── ARFIMA tsCV | h=%d | initial=%d | window=%s\n",
      horizon,cv_initial,ifelse(is.null(cv_window),"expanding",cv_window)))

    fc_fun_arfima <- local({
      dr_ <- drange; arm_ <- ar_max; mam_ <- ma_max
      function(y_tr, h, ...) {
        m_i <- tryCatch(forecast::arfima(y_tr,drange=dr_,estim="mle"),
                        error=function(e) NULL)
        if(is.null(m_i)) return(forecast::forecast(forecast::rwf(y_tr),h=h))
        tryCatch(forecast::forecast(m_i,h=h),
                 error=function(e) forecast::forecast(forecast::rwf(y_tr),h=h))
      }
    })

    e_mat <- tryCatch(
      forecast::tsCV(y=ts_obj,forecastfunction=fc_fun_arfima,
                     h=horizon,initial=cv_initial,window=cv_window),
      error=function(e){warning(sprintf("ARFIMA tsCV failed: %s",e$message));NULL})

    if(!is.null(e_mat)){
      by_h    <- .tscv_metrics(e_mat,as.numeric(ts_obj))
      overall <- tibble(RMSE=mean(by_h$RMSE,na.rm=TRUE),MAE=mean(by_h$MAE,na.rm=TRUE),
        MAPE=mean(by_h$MAPE,na.rm=TRUE),SMAPE=mean(by_h$SMAPE,na.rm=TRUE),
        MBE=mean(by_h$MBE,na.rm=TRUE))
      oos_tbl <- .tscv_oos_tibble(e_mat,ts_obj,df$date,"arfima")
      cv_results <- list(cv_metrics=overall,by_horizon=by_h,oos_tbl=oos_tbl,e_matrix=e_mat)
      if(verbose>0L){ cat("\n── ARFIMA tsCV Metrics ──\n"); print(overall) }
    }
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic()
    n_total <- nrow(df); n_ho <- max(horizon,floor(n_total*cv_holdout_frac))
    n_tr <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    cat(sprintf("\n── ARFIMA hold-out (train=%d,test=%d) ──\n",n_tr,nrow(tst_df)))
    ts_tr <- ts(tr_df$value,frequency=freq,start=c(as.integer(format(min(tr_df$date),"%Y")),
                  if(freq==12L) as.integer(format(min(tr_df$date),"%m")) else 1L))
    m_ho  <- tryCatch(forecast::arfima(ts_tr,drange=drange,estim="mle"),error=function(e) NULL)
    n_tst <- nrow(tst_df); fv <- numeric(n_tst)
    if(!is.null(m_ho)){
      all_ts <- ts(df$value,frequency=freq,start=c(as.integer(format(min(df$date),"%Y")),
                     if(freq==12L) as.integer(format(min(df$date),"%m")) else 1L))
      refit  <- tryCatch(forecast::Arima(all_ts,model=m_ho),error=function(e) NULL)
      if(!is.null(refit)){af<-as.numeric(fitted(refit));fv<-af[(n_tr+1L):(n_tr+n_tst)]}
      else fv <- tryCatch(as.numeric(forecast::forecast(m_ho,h=n_tst)$mean),
                           error=function(e) rep(NA_real_,n_tst))
    } else fv <- rep(NA_real_,n_tst)
    m_ho <- .safe_metrics(tst_df$value,fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f\n",m_ho$RMSE,m_ho$MAE))
    holdout_results <- tibble(fold=1L,h=seq_len(n_tst),date=tst_df$date,
      actual=tst_df$value,fitted=fv,residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec,eval_sec=eval_sec,total_sec=.toc(total_t0))
  cat(sprintf("\n✓ ARFIMA|total=%s|d=%.4f\n",.fmt_elapsed(runtime$total_sec),d_hat))
  structure(list(model=model,fd_fit=fd_fit,d_hat=d_hat,forecast=forecast_df,
    cv_results=cv_results,holdout_results=holdout_results,stationarity=stat,
    data=df,horizon=horizon,level=level,ts_obj=ts_obj,freq=freq,runtime=runtime),
    class="auto_arfima_ts")
}


# ================================================================
# ██  PART 14 — AUTO-HAR-RV  (with forecast::tsCV)
# ================================================================

#' @export
auto_har_rv <- function(data,date_col="date",rv_col="value",
                         horizon=1L,variant=c("standard","harj","harq","harcj"),
                         d_lags=1L,w_lags=5L,m_lags=22L,
                         jump_col=NULL,rq_col=NULL,log_transform=FALSE,
                         run_cv=TRUE,cv_initial=NULL,cv_window=NULL,
                         cv_holdout_frac=0.20,seed=42L,verbose=1L){
  total_t0 <- .tic(); set.seed(seed); variant <- match.arg(variant)
  df <- data%>%rename(date=!!sym(date_col),value=!!sym(rv_col))%>%
    arrange(date)%>%select(date,value,any_of(c(jump_col,rq_col)))
  if(anyNA(df$value)){warning("Missing→interp.");df$value<-zoo::na.approx(df$value,na.rm=FALSE)}
  rv_raw <- df$value; rv_use <- if(log_transform) log(pmax(rv_raw,1e-10)) else rv_raw

  .build_feats <- function(rv,jmp=NULL,rq=NULL){
    n <- length(rv)
    lag1 <- c(rep(NA_real_,d_lags),rv[seq_len(n-d_lags)])
    lagw <- c(rep(NA_real_,w_lags),zoo::rollmean(rv,w_lags,align="right",fill=NA)[seq_len(n-w_lags)])
    lagm <- c(rep(NA_real_,m_lags),zoo::rollmean(rv,m_lags,align="right",fill=NA)[seq_len(n-m_lags)])
    X <- data.frame(RV_d=lag1,RV_w=lagw,RV_m=lagm)
    if(!is.null(jmp)&&variant%in%c("harj","harcj")){
      cont <- pmax(rv-jmp,0)
      lag1_c <- c(rep(NA_real_,d_lags),cont[seq_len(n-d_lags)])
      lag1_j <- c(rep(NA_real_,d_lags),jmp[seq_len(n-d_lags)])
      if(variant=="harj") X$J_d <- lag1_j
      else {X$RV_d <- NULL; X$C_d <- lag1_c; X$J_d <- lag1_j}
    }
    if(!is.null(rq)&&variant=="harq"){
      lagq <- c(rep(NA_real_,d_lags),sqrt(rq[seq_len(n-d_lags)]))
      X$RQ_d <- lagq}
    X
  }
  jmp   <- if(!is.null(jump_col)&&jump_col%in%names(df)) df[[jump_col]] else NULL
  rq    <- if(!is.null(rq_col)&&rq_col%in%names(df)) df[[rq_col]] else NULL
  feats <- .build_feats(rv_use,jmp,rq)
  har_df <- na.omit(data.frame(y=rv_use,feats,date=df$date))
  fml   <- as.formula(paste("y ~",paste(names(feats),collapse=" + ")))

  fit_t0 <- .tic()
  model  <- lm(fml,data=har_df); fit_sec <- .toc(fit_t0)
  if(verbose>0L){cat(sprintf("\n── HAR-RV [%s] n=%d log=%s\n",variant,nrow(har_df),log_transform));print(summary(model))}

  freq_str <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date),by=freq_str,length.out=horizon+1L)[-1L]
  rv_hist   <- tail(rv_use,max(m_lags,60L))
  har_fc <- function(mdl,rv_h,h,lt){
    preds <- numeric(h)
    for(step in seq_len(h)){
      n_h  <- length(rv_h)
      lag1 <- rv_h[n_h]; lagw <- mean(rv_h[max(1,n_h-w_lags+1):n_h])
      lagm <- mean(rv_h[max(1,n_h-m_lags+1):n_h])
      nd   <- data.frame(RV_d=lag1,RV_w=lagw,RV_m=lagm)
      for(nm in setdiff(names(feats),c("RV_d","RV_w","RV_m","C_d"))) nd[[nm]] <- mean(feats[[nm]],na.rm=TRUE)
      if("C_d"%in%names(feats)) nd$C_d <- lag1
      p <- predict(mdl,newdata=nd)
      preds[step] <- if(lt) exp(p) else p
      rv_h <- c(rv_h,if(lt) p else preds[step])
    }
    preds
  }
  fc_vals <- har_fc(model,rv_hist,horizon,log_transform)
  forecast_df <- tibble(date=fut_dates,forecast=fc_vals,lower_95=NA_real_,upper_95=NA_real_)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0

  if(run_cv){
    ev_t0 <- .tic()
    ts_y  <- ts(rv_use,frequency=.ts_freq(df$date))
    if(is.null(cv_initial)) cv_initial <- max(as.integer(length(rv_use)*0.5),m_lags+30L)
    cat(sprintf("\n── HAR-RV tsCV | h=%d | initial=%d\n",horizon,cv_initial))

    fc_fun_har <- local({
      fml_ <- fml; d_=d_lags; w_=w_lags; m_=m_lags; lt_=log_transform
      function(y_tr, h, ...) {
        y_r   <- if(lt_) log(pmax(as.numeric(y_tr),1e-10)) else as.numeric(y_tr)
        n     <- length(y_r)
        lag1  <- c(rep(NA_real_,d_),y_r[seq_len(n-d_)])
        lagw  <- c(rep(NA_real_,w_),zoo::rollmean(y_r,w_,align="right",fill=NA)[seq_len(n-w_)])
        lagm  <- c(rep(NA_real_,m_),zoo::rollmean(y_r,m_,align="right",fill=NA)[seq_len(n-m_)])
        df_tr <- na.omit(data.frame(y=y_r,RV_d=lag1,RV_w=lagw,RV_m=lagm))
        m_i   <- tryCatch(lm(y~RV_d+RV_w+RV_m,data=df_tr),error=function(e) NULL)
        if(is.null(m_i)||nrow(df_tr)<5L) return(forecast::forecast(forecast::rwf(y_tr),h=h))
        rv_h  <- tail(y_r,max(m_,60L))
        preds <- numeric(h)
        for(step in seq_len(h)){
          n_h  <- length(rv_h)
          nd   <- data.frame(RV_d=rv_h[n_h],RV_w=mean(rv_h[max(1,n_h-w_+1):n_h]),
                              RV_m=mean(rv_h[max(1,n_h-m_+1):n_h]))
          p    <- predict(m_i,newdata=nd)
          preds[step] <- if(lt_) exp(p) else p
          rv_h <- c(rv_h,if(lt_) p else preds[step])
        }
        structure(list(mean=ts(preds,start=length(y_tr)+1L,frequency=frequency(y_tr))),class="forecast")
      }
    })

    e_mat <- tryCatch(
      forecast::tsCV(y=ts_y,forecastfunction=fc_fun_har,
                     h=horizon,initial=cv_initial,window=cv_window),
      error=function(e){warning(sprintf("HAR tsCV failed: %s",e$message));NULL})

    if(!is.null(e_mat)){
      by_h    <- .tscv_metrics(e_mat,as.numeric(ts_y))
      overall <- tibble(RMSE=mean(by_h$RMSE,na.rm=TRUE),MAE=mean(by_h$MAE,na.rm=TRUE),
        MAPE=mean(by_h$MAPE,na.rm=TRUE),SMAPE=mean(by_h$SMAPE,na.rm=TRUE),
        MBE=mean(by_h$MBE,na.rm=TRUE))
      oos_tbl <- .tscv_oos_tibble(e_mat,ts_y,df$date[seq_along(rv_use)],"har")
      cv_results <- list(cv_metrics=overall,by_horizon=by_h,oos_tbl=oos_tbl,e_matrix=e_mat)
      if(verbose>0L){ cat("\n── HAR-RV tsCV Metrics ──\n"); print(overall) }
    }
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0   <- .tic()
    n_total <- nrow(df); n_ho <- max(horizon,floor(n_total*cv_holdout_frac))
    n_tr    <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    cat(sprintf("\n── HAR-RV hold-out (train=%d,test=%d) ──\n",n_tr,nrow(tst_df)))
    rv_tr <- if(log_transform) log(pmax(tr_df$value,1e-10)) else tr_df$value
    f_tr  <- .build_feats(rv_tr,NULL,NULL); df_tr <- na.omit(data.frame(y=rv_tr,f_tr))
    m_ho  <- tryCatch(lm(fml,data=df_tr),error=function(e) NULL)
    fv    <- if(!is.null(m_ho)) har_fc(m_ho,tail(rv_tr,max(m_lags,60L)),nrow(tst_df),log_transform)
             else rep(NA_real_,nrow(tst_df))
    m_ho  <- .safe_metrics(tst_df$value,fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f\n",m_ho$RMSE,m_ho$MAE))
    holdout_results <- tibble(fold=1L,h=seq_len(nrow(tst_df)),date=tst_df$date,
      actual=tst_df$value,fitted=fv,residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec,eval_sec=eval_sec,total_sec=.toc(total_t0))
  cat(sprintf("\n✓ HAR-RV|variant=%s|total=%s\n",variant,.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model,variant=variant,forecast=forecast_df,
    cv_results=cv_results,holdout_results=holdout_results,
    data=df,horizon=horizon,log_transform=log_transform,
    d_lags=d_lags,w_lags=w_lags,m_lags=m_lags,runtime=runtime),class="auto_har_rv")
}


# ================================================================
# ██  PART 15 — AUTO-GARCH  (with forecast::tsCV)
# ================================================================

#' @export
auto_garch <- function(data,date_col="date",value_col="value",
                        horizon=10L,
                        model=c("sGARCH","eGARCH","gjrGARCH","iGARCH","csGARCH"),
                        p=1L,q=1L,mean_model=c("ARMA","zero","AR","MA"),
                        arma_order=c(0L,0L),distribution="norm",
                        external_regressors=NULL,
                        run_cv=TRUE,cv_initial=NULL,cv_window=NULL,
                        cv_holdout_frac=0.20,seed=42L,verbose=1L){
  total_t0 <- .tic(); set.seed(seed)
  model_type <- match.arg(model); mean_m <- match.arg(mean_model)
  df <- data%>%rename(date=!!sym(date_col),value=!!sym(value_col))%>%
    arrange(date)%>%select(date,value)
  if(anyNA(df$value)){warning("Missing→interp.");df$value<-zoo::na.approx(df$value,na.rm=FALSE)}

  .make_spec <- function() rugarch::ugarchspec(
    variance.model=list(model=model_type,garchOrder=c(p,q)),
    mean.model=list(armaOrder=arma_order,
                    include.mean=if(mean_m=="zero") FALSE else TRUE),
    distribution.model=distribution)

  fit_t0 <- .tic()
  spec   <- .make_spec()
  model_fit <- tryCatch(
    rugarch::ugarchfit(spec=spec,data=df$value,solver="hybrid",
                       solver.control=list(trace=0)),
    error=function(e){
      warning(sprintf("GARCH hybrid failed. Trying solnp: %s",e$message))
      tryCatch(rugarch::ugarchfit(spec=spec,data=df$value,solver="solnp"),
               error=function(e2) stop(sprintf("GARCH fit failed: %s",e2$message)))})
  fit_sec <- .toc(fit_t0)
  if(verbose>0L){
    cat(sprintf("\n── GARCH[%s(%d,%d)] dist=%s n=%d\n",model_type,p,q,distribution,nrow(df)))
    ic <- rugarch::infocriteria(model_fit)
    cat(sprintf("  AIC=%.4f|BIC=%.4f\n",ic[1],ic[2]))}

  freq_str  <- infer_frequency(df$date)
  fut_dates <- seq(max(df$date),by=freq_str,length.out=horizon+1L)[-1L]
  fc_obj    <- rugarch::ugarchforecast(model_fit,n.ahead=horizon)
  sigma_fc  <- as.numeric(rugarch::sigma(fc_obj))
  mean_fc   <- as.numeric(rugarch::fitted(fc_obj))
  forecast_df <- tibble(date=fut_dates,forecast=mean_fc,sigma=sigma_fc,
    variance=sigma_fc^2,lower_95=mean_fc-1.96*sigma_fc,upper_95=mean_fc+1.96*sigma_fc)

  cv_results <- NULL; holdout_results <- NULL; eval_sec <- 0

  if(run_cv){
    ev_t0 <- .tic()
    ts_y  <- ts(df$value,frequency=.ts_freq(df$date))
    if(is.null(cv_initial)) cv_initial <- max(as.integer(nrow(df)*0.5),100L)
    cat(sprintf("\n── GARCH tsCV | h=%d | initial=%d | window=%s\n",
      horizon,cv_initial,ifelse(is.null(cv_window),"expanding",cv_window)))

    fc_fun_garch <- local({
      mt_=model_type; p_=p; q_=q; ao_=arma_order; dist_=distribution; mm_=mean_m
      function(y_tr, h, ...) {
        sp <- rugarch::ugarchspec(
          variance.model=list(model=mt_,garchOrder=c(p_,q_)),
          mean.model=list(armaOrder=ao_,include.mean=if(mm_=="zero") FALSE else TRUE),
          distribution.model=dist_)
        m_i <- tryCatch(rugarch::ugarchfit(sp,data=as.numeric(y_tr),solver="hybrid"),
                        error=function(e) NULL)
        if(is.null(m_i)) return(forecast::forecast(forecast::rwf(y_tr),h=h))
        fc_i <- tryCatch(rugarch::ugarchforecast(m_i,n.ahead=h),error=function(e) NULL)
        if(is.null(fc_i)) return(forecast::forecast(forecast::rwf(y_tr),h=h))
        mean_i <- as.numeric(rugarch::fitted(fc_i))
        structure(list(mean=ts(mean_i,start=length(y_tr)+1L,frequency=frequency(y_tr))),
                  class="forecast")
      }
    })

    e_mat <- tryCatch(
      forecast::tsCV(y=ts_y,forecastfunction=fc_fun_garch,
                     h=horizon,initial=cv_initial,window=cv_window),
      error=function(e){warning(sprintf("GARCH tsCV failed: %s",e$message));NULL})

    if(!is.null(e_mat)){
      by_h    <- .tscv_metrics(e_mat,as.numeric(ts_y))
      overall <- tibble(RMSE=mean(by_h$RMSE,na.rm=TRUE),MAE=mean(by_h$MAE,na.rm=TRUE),
        MAPE=mean(by_h$MAPE,na.rm=TRUE),SMAPE=mean(by_h$SMAPE,na.rm=TRUE),
        MBE=mean(by_h$MBE,na.rm=TRUE))
      oos_tbl <- .tscv_oos_tibble(e_mat,ts_y,df$date,"garch")
      cv_results <- list(cv_metrics=overall,by_horizon=by_h,oos_tbl=oos_tbl,e_matrix=e_mat)
      if(verbose>0L){ cat("\n── GARCH tsCV Metrics ──\n"); print(overall) }
    }
    eval_sec <- .toc(ev_t0)
  } else {
    ev_t0 <- .tic()
    n_total <- nrow(df); n_ho <- max(horizon,floor(n_total*cv_holdout_frac))
    n_tr    <- n_total-n_ho; tr_df <- df[seq_len(n_tr),]; tst_df <- df[(n_tr+1L):n_total,]
    cat(sprintf("\n── GARCH hold-out (train=%d,test=%d) ──\n",n_tr,nrow(tst_df)))
    m_ho  <- tryCatch(rugarch::ugarchfit(spec,.make_spec(),data=tr_df$value,solver="hybrid"),
                      error=function(e) NULL)
    n_tst <- nrow(tst_df)
    fv    <- if(!is.null(m_ho)){
      fc_ho <- tryCatch(rugarch::ugarchforecast(m_ho,n.ahead=min(horizon,n_tst)),error=function(e) NULL)
      if(!is.null(fc_ho)) c(as.numeric(rugarch::fitted(fc_ho)),rep(NA_real_,max(0,n_tst-horizon)))
      else rep(NA_real_,n_tst)
    } else rep(NA_real_,n_tst)
    m_ho <- .safe_metrics(tst_df$value,fv)
    cat(sprintf("  RMSE=%.4f|MAE=%.4f\n",m_ho$RMSE,m_ho$MAE))
    holdout_results <- tibble(fold=1L,h=seq_len(n_tst),date=tst_df$date,
      actual=tst_df$value,fitted=fv,residual=tst_df$value-fv)
    eval_sec <- .toc(ev_t0)
  }
  runtime <- list(fit_sec=fit_sec,eval_sec=eval_sec,total_sec=.toc(total_t0))
  cat(sprintf("\n✓ GARCH[%s]|total=%s\n",model_type,.fmt_elapsed(runtime$total_sec)))
  structure(list(model=model_fit,spec=spec,model_type=model_type,p=p,q=q,
    distribution=distribution,forecast=forecast_df,
    cv_results=cv_results,holdout_results=holdout_results,
    data=df,horizon=horizon,runtime=runtime),class="auto_garch")
}

# ================================================================
# ██  PART 16 — SHARED OOS HELPERS
# ================================================================

.get_oos <- function(obj, strategy="recursive") {
  .rn <- function(df) if("fitted"%in%names(df)&&!"predicted"%in%names(df)) rename(df,predicted=fitted) else df

  if(inherits(obj,"auto_rnn")) {
    # tsCV path
    if(!is.null(obj$cv_results)&&!is.null(obj$cv_results$oos_tbl)) {
      oos <- obj$cv_results$oos_tbl
      if("strategy"%in%names(oos)&&strategy!="both")
        oos <- dplyr::filter(oos,strategy==!!strategy)
      return(oos)
    }
    # hold-out path
    if(!is.null(obj$holdout_results)){
      rows <- list()
      for(s in c("recursive","direct","nbeats")){
        tbl <- obj$holdout_results[[s]]
        if(!is.null(tbl)){tbl <- .rn(tbl); if(strategy=="both"||strategy==s) rows[[s]] <- tbl}
      }
      return(dplyr::bind_rows(rows))
    }
  }
  if(inherits(obj,c("auto_arima_ts","auto_arfima_ts","auto_har_rv","auto_garch"))) {
    if(!is.null(obj$cv_results)&&!is.null(obj$cv_results$oos_tbl))
      return(.rn(obj$cv_results$oos_tbl))
    if(!is.null(obj$holdout_results)) return(.rn(obj$holdout_results))
  }
  stop("No OOS results found. Run with run_cv=TRUE or run_cv=FALSE.")
}

fitted_oos <- function(object,...) UseMethod("fitted_oos")
#' @export
fitted_oos.auto_rnn       <- function(object,strategy=c("both","recursive","direct"),...) .get_oos(object,match.arg(strategy))
#' @export
fitted_oos.auto_arima_ts  <- function(object,...) .get_oos(object)
#' @export
fitted_oos.auto_arfima_ts <- function(object,...) .get_oos(object)
#' @export
fitted_oos.auto_har_rv    <- function(object,...) .get_oos(object)
#' @export
fitted_oos.auto_garch     <- function(object,...) .get_oos(object)

predicted <- function(object,...) UseMethod("predicted")
.predicted_default <- function(object,aggregate=FALSE,...){
  oos <- .get_oos(object)
  if(!aggregate) return(oos)
  has_s <- "strategy"%in%names(oos)
  oos%>%group_by(across(all_of(c("h",if(has_s)"strategy" else NULL))))%>%
    summarise(mean_actual=mean(actual),mean_predicted=mean(predicted),
              RMSE=sqrt(mean(residual^2)),MAE=mean(abs(residual)),.groups="drop")
}
#' @export
predicted.auto_rnn <- function(object,strategy=c("both","recursive","direct"),aggregate=FALSE,...){
  oos <- .get_oos(object,match.arg(strategy))
  if(!aggregate) return(oos)
  has_s <- "strategy"%in%names(oos)
  oos%>%group_by(across(all_of(c("h",if(has_s)"strategy" else NULL))))%>%
    summarise(mean_actual=mean(actual),mean_predicted=mean(predicted),
              RMSE=sqrt(mean(residual^2)),MAE=mean(abs(residual)),.groups="drop")
}
#' @export
predicted.auto_arima_ts  <- .predicted_default
#' @export
predicted.auto_arfima_ts <- .predicted_default
#' @export
predicted.auto_har_rv    <- .predicted_default
#' @export
predicted.auto_garch     <- .predicted_default

cv_performance <- function(object,...) UseMethod("cv_performance")
#' @export
cv_performance.auto_rnn <- function(object,strategy=c("both","recursive","direct"),by_horizon=TRUE,...){
  strategy <- match.arg(strategy); oos <- .get_oos(object,strategy)
  has_s <- "strategy"%in%names(oos)&&dplyr::n_distinct(oos$strategy)>1
  pc    <- if("fitted"%in%names(oos))"fitted" else "predicted"
  per_h <- if(has_s) oos%>%group_by(h,strategy) else oos%>%group_by(h)
  per_h <- per_h%>%summarise(
    RMSE=sqrt(mean(residual^2,na.rm=TRUE)),MAE=mean(abs(residual),na.rm=TRUE),
    MAPE=mean(abs(residual/actual)*100,na.rm=TRUE),
    SMAPE=mean(200*abs(residual)/(abs(actual)+abs(.data[[pc]])),na.rm=TRUE),
    MBE=mean(residual,na.rm=TRUE),.groups="drop")
  cat("\n── OOS CV Performance (tsCV) ──\n"); print(per_h,n=Inf)
  if(!is.null(object$cv_results)&&!is.null(object$cv_results$cv_metrics)){
    cat("\n── Overall tsCV Metrics ──\n"); print(object$cv_results$cv_metrics)
    cat("\n── By Horizon ──\n"); print(object$cv_results$by_horizon,n=Inf)
  }
  invisible(list(by_horizon=per_h,overall=object$cv_results$cv_metrics,raw_oos=oos))
}
#' @export
cv_performance.auto_arima_ts <- function(object,...){
  cat("\n── ARIMA tsCV Performance ──\n")
  if(!is.null(object$cv_results)){
    cat("Overall:\n"); print(object$cv_results$cv_metrics)
    cat("By horizon:\n"); print(object$cv_results$by_horizon,n=Inf)
  }
  invisible(list(overall=object$cv_results$cv_metrics,
                 by_horizon=object$cv_results$by_horizon,
                 raw_oos=.get_oos(object)))
}
#' @export
cv_performance.auto_arfima_ts <- cv_performance.auto_arima_ts
#' @export
cv_performance.auto_har_rv    <- cv_performance.auto_arima_ts
#' @export
cv_performance.auto_garch     <- cv_performance.auto_arima_ts


# ================================================================
# ██  PART 17 — S3 GENERICS (print / summary / fitted / residuals / predict / coef)
# ================================================================

#' @export
print.auto_rnn <- function(x,digits=4L,n_fc=5L,...){
  mt <- toupper(x$model_type); rt <- x$runtime
  cat("\n",.box(),"\n  Auto",mt," (torchforecast v4)\n",.box(),"\n",sep="")
  cat(.hdr("Data"),"\n")
  cat(sprintf("  Obs: %d | %s→%s | freq=%s\n",nrow(x$data),
              format(min(x$data$date)),format(max(x$data$date)),infer_frequency(x$data$date)))
  cat(.hdr("Model"),"\n")
  cat(sprintf("  arch=%s|strategy=%s|h=%d|lags=%d\n",mt,x$strategy,x$horizon,x$lags))
  hp <- x$best_params
  if(!is.null(hp)) cat(sprintf("  HPs: hidden=%d|layers=%d|dropout=%.3f|dense=%d|lr=%.2e|batch=%d\n",
    hp$hidden_size,hp$num_layers,hp$dropout,hp$dense_units,hp$lr,hp$batch_size))
  cat(.hdr("Evaluation (forecast::tsCV)"),"\n")
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$cv_metrics)){
    cat("  Overall tsCV metrics:\n"); print(x$cv_results$cv_metrics,n=Inf)
  } else if(!is.null(x$holdout_results))
    cat(sprintf("  Hold-out | train=%d | test=%d\n",
                x$holdout_results$n_train%||%"?",x$holdout_results$n_test%||%"?"))
  cat(.hdr("Runtime"),"\n")
  cat(sprintf("  BO=%s | Eval=%s | Final=%s | TOTAL=%s\n",
    .fmt_elapsed(rt$bo_sec%||%0),
    .fmt_elapsed((rt$cv_sec%||%rt$holdout_sec)%||%0),
    .fmt_elapsed(rt$final_train_sec%||%0),
    .fmt_elapsed(rt$total_sec%||%0)))
  fc_list <- Filter(Negate(is.null),list(x$forecast_recursive,x$forecast_direct))
  if(length(fc_list)>0){
    cat(.hdr("Forecast (head)"),"\n")
    for(fc in fc_list){cat(sprintf("  [%s]\n",toupper(fc$strategy[1])))
      for(i in seq_len(min(n_fc,nrow(fc))))
        cat(sprintf("    %-12s fc=%s [%s, %s]\n",format(fc$date[i]),
          .fmt(fc$forecast[i],digits),.fmt(fc$lower_95[i],digits),.fmt(fc$upper_95[i],digits)))}}
  cat(.box(),"\n\n"); invisible(x)
}

#' @export
print.auto_arima_ts <- function(x,digits=4L,...){
  m <- x$model; rt <- x$runtime
  cat("\n",.box(),"\n  Auto-ARIMA (torchforecast v4 | tsCV)\n",.box(),"\n",sep="")
  cat(sprintf("  ARIMA(%d,%d,%d)",m$arma[1],m$arma[6],m$arma[2]))
  if(x$freq>1L) cat(sprintf("(%d,%d,%d)[%d]",m$arma[3],m$arma[7],m$arma[4],x$freq))
  cat(sprintf(" | AICc=%.2f\n",m$aicc))
  if(!is.null(x$cv_results)){cat("  tsCV metrics:\n"); print(x$cv_results$cv_metrics)}
  cat(sprintf("  Fit=%s | Eval=%s | TOTAL=%s\n",
    .fmt_elapsed(rt$fit_sec),.fmt_elapsed(rt$eval_sec),.fmt_elapsed(rt$total_sec)))
  cat(.box(),"\n\n"); invisible(x)
}

#' @export
print.auto_arfima_ts <- function(x,digits=4L,...){
  rt <- x$runtime
  cat("\n",.box(),"\n  Auto-ARFIMA (torchforecast v4 | tsCV)\n",.box(),"\n",sep="")
  cat(sprintf("  d=%.6f | Fit=%s | Eval=%s | TOTAL=%s\n",
    x$d_hat,.fmt_elapsed(rt$fit_sec),.fmt_elapsed(rt$eval_sec),.fmt_elapsed(rt$total_sec)))
  if(!is.null(x$cv_results)){cat("  tsCV metrics:\n"); print(x$cv_results$cv_metrics)}
  cat(.box(),"\n\n"); invisible(x)
}

#' @export
print.auto_har_rv <- function(x,...){
  rt <- x$runtime
  cat("\n",.box(),"\n  HAR-RV (torchforecast v4 | tsCV)\n",.box(),"\n",sep="")
  cat(sprintf("  variant=%s|log=%s|lags:d=%d w=%d m=%d\n",
              x$variant,x$log_transform,x$d_lags,x$w_lags,x$m_lags))
  if(!is.null(x$cv_results)){cat("  tsCV metrics:\n"); print(x$cv_results$cv_metrics)}
  cat(sprintf("  TOTAL=%s\n",.fmt_elapsed(rt$total_sec)))
  cat(.box(),"\n\n"); invisible(x)
}

#' @export
print.auto_garch <- function(x,...){
  rt <- x$runtime; ic <- rugarch::infocriteria(x$model)
  cat("\n",.box(),"\n  GARCH (torchforecast v4 | tsCV)\n",.box(),"\n",sep="")
  cat(sprintf("  %s(%d,%d)|dist=%s|AIC=%.4f|BIC=%.4f\n",
              x$model_type,x$p,x$q,x$distribution,ic[1],ic[2]))
  if(!is.null(x$cv_results)){cat("  tsCV metrics:\n"); print(x$cv_results$cv_metrics)}
  cat(sprintf("  TOTAL=%s\n",.fmt_elapsed(rt$total_sec)))
  cat(.box(),"\n\n"); invisible(x)
}

#' @export
summary.auto_rnn        <- function(object,...){ print(object,...); invisible(object) }
#' @export
summary.auto_arima_ts   <- function(object,...){ print(object,...); print(summary(object$model)); invisible(object) }
#' @export
summary.auto_arfima_ts  <- function(object,...){
  print(object,...)
  tryCatch(print(summary(object$model)),error=function(e) print(summary(object$fd_fit)))
  invisible(object)
}
#' @export
summary.auto_har_rv     <- function(object,...){ print(object,...); print(summary(object$model)); invisible(object) }
#' @export
summary.auto_garch      <- function(object,...){ print(object,...); rugarch::show(object$model); invisible(object) }

#' @export
coef.auto_rnn <- function(object,...) unlist(lapply(object$best_params,as.numeric))

#' @export
fitted.auto_rnn <- function(object,...){
  if(is.null(object$model_recursive)) stop("No recursive model.")
  is_nb <- object$model_type=="nbeats"
  sv    <- make_supervised(scale_x(object$data$value,object$scaler),object$lags)
  pred  <- predict_deep(object$model_recursive,sv$X,get_device(),is_nb)
  tibble(date=object$data$date[(object$lags+1L):nrow(object$data)],
         fitted=unscale_x(pred,object$scaler),
         actual=object$data$value[(object$lags+1L):nrow(object$data)])%>%
    mutate(residual=actual-fitted)
}
#' @export
fitted.auto_arima_ts <- function(object,...){
  fv <- as.numeric(fitted(object$model)); n <- min(length(fv),nrow(object$data))
  tibble(date=object$data$date[seq_len(n)],fitted=fv[seq_len(n)],
         actual=object$data$value[seq_len(n)])%>%mutate(residual=actual-fitted)
}
#' @export
fitted.auto_arfima_ts <- function(object,...){
  fv <- tryCatch(as.numeric(fitted(object$model)),error=function(e) rep(NA_real_,nrow(object$data)))
  n  <- min(length(fv),nrow(object$data))
  tibble(date=object$data$date[seq_len(n)],fitted=fv[seq_len(n)],
         actual=object$data$value[seq_len(n)])%>%mutate(residual=actual-fitted)
}
#' @export
fitted.auto_har_rv <- function(object,...){
  fv <- as.numeric(fitted(object$model))
  tibble(date=object$data$date[seq_along(fv)],
         fitted=if(object$log_transform) exp(fv) else fv,
         actual=object$data$value[seq_along(fv)])%>%mutate(residual=actual-fitted)
}
#' @export
fitted.auto_garch <- function(object,...){
  fv <- as.numeric(rugarch::fitted(object$model))
  sv <- as.numeric(rugarch::sigma(object$model))
  tibble(date=object$data$date[seq_along(fv)],fitted=fv,sigma=sv,
         actual=object$data$value[seq_along(fv)])%>%mutate(residual=actual-fitted)
}

#' @export
residuals.auto_rnn        <- function(object,...) fitted(object)$residual
#' @export
residuals.auto_arima_ts   <- function(object,...) as.numeric(residuals(object$model))
#' @export
residuals.auto_arfima_ts  <- function(object,...) tryCatch(as.numeric(residuals(object$model)),
                               error=function(e) as.numeric(residuals(object$fd_fit)))
#' @export
residuals.auto_har_rv     <- function(object,...) as.numeric(residuals(object$model))
#' @export
residuals.auto_garch      <- function(object,standardized=TRUE,...){
  if(standardized) as.numeric(rugarch::residuals(object$model,standardize=TRUE))
  else             as.numeric(rugarch::residuals(object$model))
}

#' @export
predict.auto_rnn <- function(object,newdata=NULL,horizon=NULL,
                              strategy=c("both","recursive","direct"),level=95,...){
  strategy <- match.arg(strategy); h <- horizon%||%object$horizon; sc <- object$scaler
  input_vals <- if(!is.null(newdata)) newdata$value else object$data$value
  last_date  <- if(!is.null(newdata)) max(newdata$date) else max(object$data$date)
  freq_str   <- infer_frequency(object$data$date)
  last_win   <- tail(scale_x(input_vals,sc),object$lags)
  z_val <- qnorm(0.5+level/200)
  res_sd <- tryCatch({
    sv <- make_supervised(scale_x(object$data$value,sc),object$lags)
    pd <- predict_deep(object$model_recursive,sv$X,get_device(),object$model_type=="nbeats")
    sd(unscale_x(sv$y,sc)-unscale_x(pd,sc))
  },error=function(e) sd(input_vals)*0.1)
  fut_dates <- seq(last_date,by=freq_str,length.out=h+1L)[-1L]
  ci_w <- z_val*res_sd*sqrt(seq_len(h))
  lo <- paste0("lower_",level); hi <- paste0("upper_",level)
  res <- list()
  if(strategy%in%c("recursive","both")&&!is.null(object$model_recursive)){
    fc <- if(object$model_type=="nbeats") nbeats_forecast(object$model_recursive,last_win,h,sc,get_device())
          else recursive_forecast(object$model_recursive,last_win,h,sc,get_device())
    res$recursive <- tibble(date=fut_dates,predicted=fc,!!lo:=fc-ci_w,!!hi:=fc+ci_w,
                             strategy=if(object$model_type=="nbeats")"nbeats" else "recursive")
  }
  if(strategy%in%c("direct","both")&&!is.null(object$direct_models)&&length(object$direct_models)>=h){
    fc <- direct_forecast(object$direct_models[seq_len(h)],last_win,sc,get_device())
    res$direct <- tibble(date=fut_dates,predicted=fc,!!lo:=fc-ci_w,!!hi:=fc+ci_w,strategy="direct")
  }
  if(length(res)==0L) stop("No models available for the requested strategy.")
  dplyr::bind_rows(res)
}
#' @export
predict.auto_arima_ts <- function(object,horizon=NULL,newdata=NULL,level=NULL,xreg=NULL,...){
  h  <- horizon%||%object$horizon; lv <- level%||%object$level
  m  <- if(!is.null(newdata)) tryCatch(forecast::Arima(.df_to_ts(newdata),model=object$model),
         error=function(e) object$model) else object$model
  fc <- forecast::forecast(m,h=h,level=lv,xreg=xreg)
  base_date <- if(!is.null(newdata)) max(newdata[[1]]) else max(object$data$date)
  fut_dates <- seq(base_date,by=infer_frequency(object$data$date),length.out=h+1L)[-1L]
  tibble(date=fut_dates,predicted=as.numeric(fc$mean),
         lower_95=as.numeric(fc$lower[,ncol(fc$lower)]),
         upper_95=as.numeric(fc$upper[,ncol(fc$upper)]))
}
#' @export
predict.auto_arfima_ts <- function(object,horizon=NULL,level=NULL,...){
  h  <- horizon%||%object$horizon; lv <- level%||%object$level
  fc <- tryCatch(forecast::forecast(object$model,h=h,level=lv),error=function(e) NULL)
  if(is.null(fc)) return(object$forecast[seq_len(h),])
  fut_dates <- seq(max(object$data$date),by=infer_frequency(object$data$date),length.out=h+1L)[-1L]
  tibble(date=fut_dates,predicted=as.numeric(fc$mean),
         lower_95=as.numeric(fc$lower[,ncol(fc$lower)]),
         upper_95=as.numeric(fc$upper[,ncol(fc$upper)]))
}
#' @export
predict.auto_har_rv <- function(object,horizon=NULL,...){
  h <- horizon%||%object$horizon; object$forecast[seq_len(h),]
}
#' @export
predict.auto_garch <- function(object,horizon=NULL,...){
  h  <- horizon%||%object$horizon
  fc <- rugarch::ugarchforecast(object$model,n.ahead=h)
  tibble(predicted=as.numeric(rugarch::fitted(fc)),
         sigma=as.numeric(rugarch::sigma(fc)),variance=as.numeric(rugarch::sigma(fc))^2)
}

#' @export
format.auto_rnn <- function(x,...) sprintf("<auto_%s|n=%d|h=%d|eval=%s|t=%s>",
  x$model_type,nrow(x$data),x$horizon,
  if(!is.null(x$cv_results))"tsCV" else "holdout",
  .fmt_elapsed(x$runtime$total_sec%||%NA))
#' @export
autoplot.auto_rnn       <- function(object,...){ p <- plot(object,...); invisible(p) }
#' @export
autoplot.auto_arima_ts  <- function(object,...){ p <- plot(object,...); invisible(p) }
#' @export
autoplot.auto_arfima_ts <- function(object,...){ p <- plot(object,...); invisible(p) }


# ================================================================
# ██  PART 18 — PLOT METHODS
# ================================================================

.bt <- .base_theme

.fc_panel <- function(hist_df,fc_all,label,colour,bt){
  p <- ggplot()+
    geom_line(data=hist_df,aes(date,value),colour=.PAL$actual,linewidth=0.8)+
    geom_vline(xintercept=as.numeric(max(hist_df$date)),linetype="dashed",colour=.PAL$neutral)
  if(!is.null(fc_all)&&nrow(fc_all)>0){
    sp <- c(recursive=.PAL$recursive,direct=.PAL$direct,nbeats=.PAL$nbeats,model=colour)
    p <- p+
      geom_ribbon(data=fc_all,aes(date,ymin=lower_95,ymax=upper_95,
                                   fill=if("strategy"%in%names(fc_all)) strategy else "model"),alpha=0.15)+
      geom_line(data=fc_all,aes(date,forecast,
                                 colour=if("strategy"%in%names(fc_all)) strategy else "model"),
                linewidth=1,linetype="dashed")+
      geom_point(data=fc_all,aes(date,forecast,
                                  colour=if("strategy"%in%names(fc_all)) strategy else "model"),
                 size=2,shape=21,fill="white",stroke=1.1)+
      scale_colour_manual(values=sp,name="Strategy")+scale_fill_manual(values=sp)
  }
  p+labs(title=sprintf("%s Forecast",label),subtitle="Dashed=forecast|Band=95%CI",x=NULL,y="Value")+bt()
}

#' @export
plot.auto_rnn <- function(x,which=NULL,last_n=NULL,...){
  mt <- toupper(x$model_type); panels <- list()
  hist_df <- if(!is.null(last_n)) tail(x$data,last_n) else x$data
  fc_all  <- dplyr::bind_rows(Filter(Negate(is.null),list(x$forecast_recursive,x$forecast_direct)))
  panels$forecast <- .fc_panel(hist_df,fc_all,sprintf("Auto%s",mt),.PAL[[mt]],.bt)

  if(!is.null(x$history_recursive)&&nrow(x$history_recursive)>0){
    h <- x$history_recursive; best_ep <- h$epoch[which.min(replace(h$val_loss,is.na(h$val_loss),Inf))]
    panels$training <- h%>%select(epoch,Training=train_loss,Validation=val_loss)%>%
      pivot_longer(-epoch,names_to="split",values_to="MSE")%>%filter(!is.na(MSE))%>%
      ggplot(aes(epoch,MSE,colour=split))+
      geom_vline(xintercept=best_ep,linetype="dotted",colour="grey60")+
      geom_line(linewidth=0.85)+
      scale_colour_manual(values=c(Training=.PAL$train,Validation=.PAL$val),name=NULL)+
      labs(title="Training Loss (EarlyStopping)",x="Epoch",y="MSE")+.bt()
  }
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$by_horizon)){
    bh <- x$cv_results$by_horizon
    has_s <- "strategy"%in%names(bh)&&dplyr::n_distinct(bh$strategy)>1
    panels$cv_by_h <- bh%>%select(h,if(has_s)"strategy",RMSE,MAE)%>%
      pivot_longer(c(RMSE,MAE),names_to="metric",values_to="value")%>%
      ggplot(aes(h,value,colour=if(has_s) strategy else metric,linetype=metric))+
      geom_line(linewidth=0.85)+geom_point(size=1.8,shape=21,fill="white",stroke=1)+
      scale_colour_manual(values=c(recursive=.PAL$recursive,direct=.PAL$direct,
                                   nbeats=.PAL$nbeats,RMSE=.PAL$recursive,MAE=.PAL$direct),name=NULL)+
      labs(title="tsCV Metrics by Horizon h",subtitle="Solid=RMSE|Dashed=MAE",
           x="Horizon h",y="Error")+.bt()
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
    patchwork::plot_annotation(title=sprintf("Auto%s Dashboard (tsCV)",mt),
      theme=theme(plot.title=element_text(face="bold",size=14,hjust=0.5)))
  print(pw); invisible(pw)
}

.stat_plot <- function(x,label,colour){
  panels <- list(); fc <- x$forecast; hist_df <- x$data
  fc_all <- fc%>%mutate(strategy="model")
  if(!"lower_95"%in%names(fc_all)) fc_all$lower_95 <- NA_real_
  if(!"upper_95"%in%names(fc_all)) fc_all$upper_95 <- NA_real_
  panels$forecast <- .fc_panel(hist_df,fc_all,label,colour,.bt)
  fv <- tryCatch(fitted(x),error=function(e) NULL)
  if(!is.null(fv)&&"residual"%in%names(fv))
    panels$residuals <- filter(fv,!is.na(residual))%>%ggplot(aes(date,residual))+
      geom_hline(yintercept=0,colour="grey40")+geom_line(colour=colour,linewidth=0.7,alpha=0.7)+
      labs(title="Residuals",x=NULL,y="Residual")+.bt()
  if(!is.null(x$cv_results)&&!is.null(x$cv_results$by_horizon))
    panels$by_h <- x$cv_results$by_horizon%>%
      select(h,RMSE,MAE)%>%pivot_longer(-h,names_to="metric",values_to="value")%>%
      ggplot(aes(h,value,colour=metric))+geom_line(linewidth=0.9)+
      geom_point(size=2,shape=21,fill="white",stroke=1.2)+
      scale_colour_manual(values=c(RMSE=colour,MAE=.PAL$direct),name=NULL)+
      labs(title="tsCV Error by Horizon h",x="h",y="Error")+.bt()
  pw <- patchwork::wrap_plots(panels,ncol=2L)+
    patchwork::plot_annotation(title=sprintf("%s Dashboard (tsCV)",label),
      theme=theme(plot.title=element_text(face="bold",size=14,hjust=0.5)))
  print(pw); invisible(pw)
}
#' @export
plot.auto_arima_ts  <- function(x,...) .stat_plot(x,sprintf("Auto-ARIMA(%d,%d,%d)",
  x$model$arma[1],x$model$arma[6],x$model$arma[2]),.PAL$arima)
#' @export
plot.auto_arfima_ts <- function(x,...) .stat_plot(x,sprintf("Auto-ARFIMA(d=%.4f)",x$d_hat),.PAL$arfima)
#' @export
plot.auto_har_rv    <- function(x,...) .stat_plot(x,sprintf("HAR-RV [%s]",x$variant),.PAL$har)
#' @export
plot.auto_garch     <- function(x,...) .stat_plot(x,sprintf("GARCH[%s(%d,%d)]",x$model_type,x$p,x$q),.PAL$garch)


# ================================================================
# ██  PART 19 — plot_oos_fit()  (universal OOS actual vs fitted)
# ================================================================

plot_oos_fit <- function(object,...) UseMethod("plot_oos_fit")

.oos_dash <- function(oos,hist_df,title,eval_label,model_label,ncol=2L){
  bt <- .bt()
  if("fitted"%in%names(oos)&&!"predicted"%in%names(oos)) oos <- rename(oos,predicted=fitted)
  if(!"residual"%in%names(oos)) oos <- mutate(oos,residual=actual-predicted)
  has_s    <- "strategy"%in%names(oos)&&dplyr::n_distinct(oos$strategy)>1L
  has_fold <- "fold"%in%names(oos)&&dplyr::n_distinct(oos$fold)>1L
  has_h    <- "h"%in%names(oos)
  has_date <- "date"%in%names(oos)&&!all(is.na(oos$date))
  s_pal    <- c(recursive=.PAL$recursive,direct=.PAL$direct,nbeats=.PAL$nbeats,
                model=.PAL$arima,arima=.PAL$arima,arfima=.PAL$arfima,har=.PAL$har,garch=.PAL$garch)
  panels   <- list()

  if(has_date){
    grp_v <- c("date",if(has_s)"strategy" else NULL)
    agg   <- oos%>%group_by(across(all_of(grp_v)))%>%
      summarise(mean_actual=mean(actual,na.rm=TRUE),mean_predicted=mean(predicted,na.rm=TRUE),
                sd_predicted=sd(predicted,na.rm=TRUE),.groups="drop")
    p1 <- ggplot()
    if(!is.null(hist_df)&&nrow(hist_df)>0)
      p1 <- p1+geom_line(data=hist_df,aes(date,value),colour="grey40",linewidth=0.65,alpha=0.55)+
        geom_vline(xintercept=as.numeric(min(oos$date,na.rm=TRUE)),
                   linetype="dashed",colour="grey60",linewidth=0.5)
    if(has_s&&"sd_predicted"%in%names(agg))
      p1 <- p1+geom_ribbon(data=agg,aes(date,ymin=mean_predicted-sd_predicted,
                                          ymax=mean_predicted+sd_predicted,fill=strategy),alpha=0.12)
    p1 <- p1+
      geom_line(data=agg,aes(date,mean_actual),colour="grey20",linewidth=1.0)+
      geom_line(data=agg,aes(date,mean_predicted,colour=if(has_s) strategy else "model"),
                linewidth=1.0,linetype="dashed")+
      geom_point(data=agg,aes(date,mean_predicted,colour=if(has_s) strategy else "model"),
                 size=1.8,shape=21,fill="white",stroke=1)+
      scale_colour_manual(values=s_pal,name=NULL)+scale_fill_manual(values=s_pal)+
      labs(title="Actual vs Fitted (OOS)",
           subtitle=sprintf("%s | black=actual, dashed=fitted%s",eval_label,
                            if(!is.null(agg$sd_predicted)&&any(!is.na(agg$sd_predicted)))" | band=±1SD" else ""),
           x=NULL,y="Value")+bt
    panels$time_series <- p1
  }

  panels$scatter <- ggplot(oos,aes(actual,predicted,colour=if(has_s) strategy else "model"))+
    geom_abline(slope=1,intercept=0,colour="grey40",linetype="dashed")+
    geom_point(alpha=0.5,size=1.5)+
    scale_colour_manual(values=s_pal,name=NULL)+
    labs(title="Actual vs Fitted Scatter",subtitle="Dashed=perfect fit",x="Actual",y="Fitted")+
    coord_equal()+bt

  if(has_date){
    res_agg <- oos%>%group_by(across(all_of(c("date",if(has_s)"strategy" else NULL))))%>%
      summarise(mean_res=mean(residual,na.rm=TRUE),.groups="drop")
    r_sd <- sd(oos$residual,na.rm=TRUE)
    panels$residuals_time <- ggplot(res_agg,aes(date,mean_res,colour=if(has_s) strategy else "model"))+
      geom_hline(yintercept=0,colour="grey40")+
      geom_hline(yintercept=c(-2,2)*r_sd,linetype="dashed",colour=.PAL$direct,alpha=0.7)+
      geom_line(linewidth=0.6,alpha=0.8)+geom_point(alpha=0.5,size=1.2)+
      geom_smooth(method="loess",formula=y~x,se=FALSE,linewidth=0.75,colour=.PAL$direct)+
      scale_colour_manual(values=s_pal,name=NULL)+
      labs(title="OOS Residuals over Time",subtitle="±2σ|LOESS",x=NULL,y="Residual")+bt
  }

  if(has_h){
    by_h <- oos%>%group_by(across(all_of(c("h",if(has_s)"strategy" else NULL))))%>%
      summarise(RMSE=sqrt(mean(residual^2,na.rm=TRUE)),MAE=mean(abs(residual),na.rm=TRUE),.groups="drop")%>%
      pivot_longer(c(RMSE,MAE),names_to="metric",values_to="value")
    panels$by_horizon <- ggplot(by_h,aes(h,value,colour=if(has_s) strategy else "model",linetype=metric))+
      geom_line(linewidth=0.85)+geom_point(size=1.8,shape=21,fill="white",stroke=1)+
      scale_colour_manual(values=s_pal,name=NULL)+
      scale_linetype_manual(values=c(RMSE="solid",MAE="dashed"),name="Metric")+
      labs(title="Error by Horizon Step h (tsCV)",x="h",y="Error")+bt
  }

  panels$error_dist <- ggplot(oos,aes(residual,fill=if(has_s) strategy else "model"))+
    geom_histogram(bins=30L,alpha=0.65,position="identity",colour="white")+
    geom_vline(xintercept=0,colour="grey30",linetype="dashed")+
    scale_fill_manual(values=s_pal,name=NULL)+
    labs(title="OOS Error Distribution",x="Residual",y="Count")+bt

  if(has_fold&&dplyr::n_distinct(oos$fold)>5L){
    fold_r <- oos%>%group_by(across(all_of(c("fold",if(has_s)"strategy" else NULL))))%>%
      summarise(RMSE=sqrt(mean(residual^2,na.rm=TRUE)),.groups="drop")
    panels$fold_box <- ggplot(fold_r,aes(x=if(has_s) strategy else "model",y=RMSE,
                                          fill=if(has_s) strategy else "model"))+
      geom_boxplot(alpha=0.65,outlier.shape=21,show.legend=FALSE)+
      geom_jitter(width=0.15,alpha=0.5,size=1.5,show.legend=FALSE)+
      scale_fill_manual(values=s_pal)+
      labs(title=sprintf("RMSE across tsCV Origins (%d)",dplyr::n_distinct(oos$fold)),x=NULL,y="RMSE")+bt
  }

  pw <- patchwork::wrap_plots(panels,ncol=ncol)+
    patchwork::plot_annotation(title=title,
      caption=sprintf("Engine: %s  ·  %s",eval_label,model_label),
      theme=theme(plot.title=element_text(face="bold",size=14,hjust=0.5),
                  plot.caption=element_text(size=9,colour="grey50",hjust=0.5)))
  print(pw); invisible(pw)
}

#' @export
plot_oos_fit.auto_rnn <- function(object,strategy=c("both","recursive","direct"),
                                   last_n=NULL,ncol=2L,title=NULL,...){
  strategy <- match.arg(strategy); oos <- .get_oos(object,strategy)
  if(nrow(oos)==0L) stop("Empty OOS tibble.")
  hist_df <- if(!is.null(last_n)) tail(object$data,last_n) else object$data
  if(is.null(title)) title <- sprintf("Auto%s — OOS Actual vs Fitted",toupper(object$model_type))
  ev_tag  <- if(!is.null(object$cv_results)) "forecast::tsCV" else "Hold-out test set"
  .oos_dash(oos,hist_df,title,ev_tag,toupper(object$model_type),ncol)
}
#' @export
plot_oos_fit.auto_arima_ts <- function(object,last_n=NULL,ncol=2L,title=NULL,...){
  oos <- .get_oos(object); hist_df <- if(!is.null(last_n)) tail(object$data,last_n) else object$data
  ord <- object$model$arma
  if(is.null(title)) title <- sprintf("Auto-ARIMA(%d,%d,%d) — OOS Actual vs Fitted",ord[1],ord[6],ord[2])
  .oos_dash(oos,hist_df,title,if(!is.null(object$cv_results))"forecast::tsCV" else "Hold-out","ARIMA",ncol)
}
#' @export
plot_oos_fit.auto_arfima_ts <- function(object,last_n=NULL,ncol=2L,title=NULL,...){
  oos <- .get_oos(object); hist_df <- if(!is.null(last_n)) tail(object$data,last_n) else object$data
  if(is.null(title)) title <- sprintf("Auto-ARFIMA(d=%.4f) — OOS Actual vs Fitted",object$d_hat)
  .oos_dash(oos,hist_df,title,if(!is.null(object$cv_results))"forecast::tsCV" else "Hold-out","ARFIMA",ncol)
}
#' @export
plot_oos_fit.auto_har_rv <- function(object,last_n=NULL,ncol=2L,title=NULL,...){
  oos <- .get_oos(object); hist_df <- if(!is.null(last_n)) tail(object$data,last_n) else object$data
  if(is.null(title)) title <- sprintf("HAR-RV[%s] — OOS Actual vs Fitted",object$variant)
  .oos_dash(oos,hist_df,title,if(!is.null(object$cv_results))"forecast::tsCV" else "Hold-out","HAR-RV",ncol)
}
#' @export
plot_oos_fit.auto_garch <- function(object,last_n=NULL,ncol=2L,title=NULL,...){
  oos <- .get_oos(object); hist_df <- if(!is.null(last_n)) tail(object$data,last_n) else object$data
  if(is.null(title)) title <- sprintf("GARCH[%s(%d,%d)] — OOS Actual vs Fitted",
                                       object$model_type,object$p,object$q)
  .oos_dash(oos,hist_df,title,if(!is.null(object$cv_results))"forecast::tsCV" else "Hold-out","GARCH",ncol)
}
#' @export
plot_oos_fit.default <- function(object,hist_df=NULL,title="OOS Actual vs Fitted",ncol=2L,...){
  if(!"actual"%in%names(object)) stop("Tibble needs 'actual' and 'predicted'/'fitted' columns.")
  .oos_dash(object,hist_df,title,"Rolling-origin","Model",ncol)
}


# ================================================================
# ██  PART 20 — DIEBOLD-MARIANO TEST
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
  if(T<10L) warning("DM test may be unreliable with T<10.")
  .L<-function(e,a) switch(loss,
    MSE=e^2,MAE=abs(e),
    MAPE={if(is.null(a)) stop("actual required"); abs(e/a)*100},
    SMAPE={if(is.null(a)) stop("actual required"); 200*abs(e)/(abs(a)+abs(a-e))},
    MBE=e,POWER=abs(e)^power,LINEX=exp(linex_a*e)-linex_a*e-1,
    custom={if(is.null(custom_fn)) stop("custom_fn required"); custom_fn(e,a)})
  d<-.L(e1,actual)-.L(e2,actual); d_bar<-mean(d)
  lm_d<-lm(d~1)
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
    less     =  (if(is.finite(df_stat)) pt(DM_adj,df=df_stat) else pnorm(DM_adj)),
    greater  =  (if(is.finite(df_stat)) pt(-DM_adj,df=df_stat) else pnorm(-DM_adj)))
  alpha<-1-conf_level; q<-if(is.finite(df_stat)) qt(1-alpha/2,df=df_stat) else qnorm(1-alpha/2)
  ci<-c(lower=d_bar-q*se,upper=d_bar+q*se)
  concl<-if(p_val>=(1-conf_level))
    sprintf("FAIL TO REJECT H0 (p=%.4f): No significant difference.",p_val)
  else sprintf("REJECT H0 (p=%.4f): %s %s",p_val,
               if(d_bar>0)"Model 1 has HIGHER loss" else "Model 1 has LOWER loss",
               dplyr::case_when(p_val<0.001~"***",p_val<0.01~"**",p_val<0.05~"*",TRUE~""))
  structure(list(statistic=DM_adj,dm_raw=DM,p_value=p_val,alternative=alternative,
    loss=loss,h=h,T=T,d_bar=d_bar,lrv=lrv*T,se=se,ci=ci,conf_level=conf_level,
    hln=hln,hln_factor=hln_f,df=df_stat,bw=bw,bw_method=bw_method,
    d_series=d,L1=.L(e1,actual),L2=.L(e2,actual),conclusion=concl),class="dm_test")
}
#' @export
print.dm_test <- function(x,...){
  cat("\n",.box(),"  Diebold-Mariano Test (tsCV-based errors)\n",.box(),"\n",sep="")
  cat(sprintf("  Loss=%-6s|h=%d|n=%d|alt=%s|HLN=%s\n",x$loss,x$h,x$T,x$alternative,x$hln))
  cat(sprintf("  d̄=%+.6f|SE=%.6f|DM*=%+.4f|p=%.6f %s\n",x$d_bar,x$se,x$statistic,x$p_value,
    dplyr::case_when(x$p_value<0.001~"***",x$p_value<0.01~"**",x$p_value<0.05~"*",x$p_value<0.10~".",TRUE~"")))
  cat(sprintf("  %.0f%% CI: [%+.6f, %+.6f]\n",x$conf_level*100,x$ci["lower"],x$ci["upper"]))
  cat(sprintf("  %s\n",x$conclusion))
  cat(.box(),"\n\n"); invisible(x)
}
#' @export
plot.dm_test <- function(x,which=NULL,...){
  bt<-.bt(); T<-x$T; panels<-list()
  ld<-tibble(t=seq_len(T),L1=x$L1,L2=x$L2)%>%pivot_longer(-t,names_to="model",values_to="loss")
  panels$loss<-ggplot(ld,aes(t,loss,colour=model))+geom_line(linewidth=0.75)+
    scale_colour_manual(values=c(L1=.PAL$recursive,L2=.PAL$direct),name=NULL)+
    labs(title=sprintf("Loss [%s]",x$loss),x="Obs",y="Loss")+bt
  d_df<-tibble(t=seq_len(T),d=x$d_series)
  panels$diff<-ggplot(d_df,aes(t,d))+
    geom_hline(yintercept=0,colour="grey40")+
    geom_hline(yintercept=x$d_bar,colour=.PAL$recursive,linetype="dashed")+
    geom_ribbon(aes(ymin=x$d_bar-qnorm(0.975)*x$se,ymax=x$d_bar+qnorm(0.975)*x$se),
                fill=.PAL$recursive,alpha=0.12)+
    geom_line(colour=.PAL$arima,linewidth=0.7)+
    labs(title="Loss Differential d_t",x="Obs",y="d")+bt
  panels$hist<-ggplot(tibble(d=x$d_series),aes(d))+
    geom_histogram(aes(y=after_stat(density)),bins=25,fill=.PAL$arima,colour="white",alpha=0.7)+
    stat_function(fun=dnorm,args=list(mean=x$d_bar,sd=sd(x$d_series)),colour=.PAL$direct,linewidth=0.9)+
    geom_vline(xintercept=0,colour="grey40",linetype="dashed")+
    labs(title="Distribution of d_t",x="d",y="Density")+bt
  n_lag<-min(20L,T%/%4L); acf_d<-acf(x$d_series,lag.max=n_lag,plot=FALSE)
  ci_acf<-qnorm(0.975)/sqrt(T)
  acf_df<-tibble(lag=as.numeric(acf_d$lag[-1]),acf=as.numeric(acf_d$acf[-1]))
  panels$acf<-ggplot(acf_df,aes(lag,acf))+
    geom_hline(yintercept=0,colour="grey30")+
    geom_hline(yintercept=c(-ci_acf,ci_acf),linetype="dashed",colour=.PAL$direct)+
    geom_segment(aes(xend=lag,yend=0,colour=abs(acf)>ci_acf),linewidth=1)+
    geom_point(aes(colour=abs(acf)>ci_acf),size=1.8)+
    scale_colour_manual(values=c("FALSE"=.PAL$recursive,"TRUE"=.PAL$direct),guide="none")+
    labs(title="ACF of d_t",x="Lag",y="ACF")+bt
  sel<-if(is.null(which)) seq_along(panels) else which
  pw<-patchwork::wrap_plots(panels[sel],ncol=2L)+
    patchwork::plot_annotation(title=sprintf("DM Test|%s|p=%.4f",x$loss,x$p_value),
      theme=theme(plot.title=element_text(face="bold",size=13,hjust=0.5)))
  print(pw); invisible(pw)
}

#' DM test from two torchforecast objects (uses tsCV OOS errors)
#' @export
dm_compare <- function(result1,result2,strategy1="recursive",strategy2="recursive",
                        loss="MSE",h=NULL,by_horizon=TRUE,alternative="two.sided",hln=TRUE,...){
  oos1<-.get_oos(result1,strategy1); oos2<-.get_oos(result2,strategy2)
  if("strategy"%in%names(oos1)) oos1<-dplyr::filter(oos1,strategy==strategy1)
  if("strategy"%in%names(oos2)) oos2<-dplyr::filter(oos2,strategy==strategy2)
  if(!is.null(h)){oos1<-dplyr::filter(oos1,.data$h==!!h);oos2<-dplyr::filter(oos2,.data$h==!!h)}
  if("fold"%in%names(oos1)&&"fold"%in%names(oos2)){
    common<-intersect(oos1$fold,oos2$fold)
    oos1<-dplyr::filter(oos1,fold%in%common)%>%dplyr::arrange(fold,h)
    oos2<-dplyr::filter(oos2,fold%in%common)%>%dplyr::arrange(fold,h)}
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
      res<-tryCatch(dm_test(d1$actual[1:n]-d1[[pc(d1)]][1:n],
                             d2$actual[1:n]-d2[[pc(d2)]][1:n],
                             h=hh,loss=loss,actual=d1$actual[1:n],
                             alternative=alternative,hln=hln,...),error=function(e) NULL)
      if(is.null(res)) return(NULL)
      tibble(h=hh,n=n,DM_stat=res$statistic,p_value=res$p_value,d_bar=res$d_bar,
             sig=dplyr::case_when(res$p_value<0.01~"***",res$p_value<0.05~"**",
                                  res$p_value<0.10~"*",TRUE~""))
    })
    dplyr::bind_rows(Filter(Negate(is.null),rows))
  },error=function(e) NULL) else NULL
  list(dm=dm_res,dm_horizon=dm_h,oos1=oos1,oos2=oos2)
}

#' Pairwise DM matrix (parallel option)
#' @export
dm_matrix <- function(oos_list,loss="MSE",alternative="two.sided",
                       hln=TRUE,h=NULL,use_parallel=TRUE,...){
  t0<-.tic(); nms<-names(oos_list)
  if(is.null(nms)||any(nms=="")) stop("oos_list must be a fully named list.")
  M<-length(nms)
  sm<-matrix(NA_real_,M,M,dimnames=list(nms,nms)); diag(sm)<-0
  pm<-matrix(NA_real_,M,M,dimnames=list(nms,nms)); diag(pm)<-1
  dm_<-matrix(NA_real_,M,M,dimnames=list(nms,nms)); diag(dm_)<-0
  pairs<-do.call(c,lapply(seq_len(M),function(i) lapply(setdiff(seq_len(M),i),function(j) c(i,j))))
  test_pair<-function(pair){
    i<-pair[1]; j<-pair[2]; d1<-oos_list[[i]]; d2<-oos_list[[j]]
    if(!is.null(h)&&"h"%in%names(d1)){d1<-dplyr::filter(d1,.data$h==!!h);d2<-dplyr::filter(d2,.data$h==!!h)}
    pc<-function(df) if("fitted"%in%names(df))"fitted" else "predicted"
    n_u<-min(nrow(d1),nrow(d2))
    if(n_u<5L) return(list(i=i,j=j,stat=NA_real_,pval=NA_real_,dbar=NA_real_))
    e1<-(d1$actual-d1[[pc(d1)]])[seq_len(n_u)]
    e2<-(d2$actual-d2[[pc(d2)]])[seq_len(n_u)]
    act<-d1$actual[seq_len(n_u)]
    res<-tryCatch(dm_test(e1,e2,h=h%||%1L,loss=loss,actual=act,alternative=alternative,hln=hln,...),
                  error=function(e) NULL)
    if(is.null(res)) return(list(i=i,j=j,stat=NA_real_,pval=NA_real_,dbar=NA_real_))
    list(i=i,j=j,stat=res$statistic,pval=res$p_value,dbar=res$d_bar)
  }
  if(use_parallel&&.is_parallel())
    results<-furrr::future_map(pairs,function(p){suppressPackageStartupMessages(library(sandwich));test_pair(p)},
      .options=furrr::furrr_options(seed=TRUE,globals=FALSE),.progress=TRUE)
  else results<-lapply(pairs,test_pair)
  for(r in results){sm[r$i,r$j]<-r$stat;pm[r$i,r$j]<-r$pval;dm_[r$i,r$j]<-r$dbar}
  cat(sprintf("\n── DM matrix (%d×%d=%d tests) in %s\n",M,M,length(pairs),.fmt_elapsed(.toc(t0))))
  structure(list(stat_matrix=sm,pval_matrix=pm,dbar_matrix=dm_,
    model_names=nms,loss=loss,alternative=alternative,hln=hln,h=h),class="dm_matrix")
}
#' @export
print.dm_matrix <- function(x,digits=3,...){
  cat("\n",.box(),"  DM Pairwise Matrix (tsCV-based)\n",.box(),"\n",sep="")
  cat(sprintf("  Loss=%s|alt=%s|HLN=%s\n",x$loss,x$alternative,x$hln))
  cat("\n  [DM Statistics]\n"); print(round(x$stat_matrix,digits))
  sig<-matrix(dplyr::case_when(is.na(as.vector(x$pval_matrix))~"NA",
    as.vector(x$pval_matrix)<0.01~"***",as.vector(x$pval_matrix)<0.05~" **",
    as.vector(x$pval_matrix)<0.10~"  *",TRUE~"   "),
    nrow=nrow(x$pval_matrix),dimnames=dimnames(x$pval_matrix))
  pf<-matrix(sprintf("%.3f%s",as.vector(x$pval_matrix),as.vector(sig)),
              nrow=nrow(x$pval_matrix),dimnames=dimnames(x$pval_matrix))
  diag(pf)<-"─"; cat("\n  [p-values]\n"); print(pf,quote=FALSE)
  cat(.box(),"\n\n"); invisible(x)
}
#' @export
plot.dm_matrix <- function(x,type=c("pvalue","statistic","dbar"),...){
  type<-match.arg(type)
  mat<-switch(type,pvalue=x$pval_matrix,statistic=x$stat_matrix,dbar=x$dbar_matrix)
  df_long<-as.data.frame(mat)%>%tibble::rownames_to_column("model_row")%>%
    pivot_longer(-model_row,names_to="model_col",values_to="value")%>%
    mutate(model_row=factor(model_row,levels=x$model_names),
           model_col=factor(model_col,levels=rev(x$model_names)),
           label=dplyr::case_when(
             model_row==model_col~"─",is.na(value)~"NA",
             type=="pvalue"~sprintf("%.3f%s",value,dplyr::case_when(
               value<0.01~"***",value<0.05~"**",value<0.10~"*",TRUE~"")),
             TRUE~sprintf("%+.3f",value)))
  fs<-switch(type,
    pvalue=scale_fill_gradient2(low="#d73027",mid="#ffffbf",high="#1a9850",
                                 midpoint=0.10,na.value="grey90",limits=c(0,1),name="p"),
    scale_fill_gradient2(low="#2166ac",mid="white",high="#d6604d",midpoint=0,na.value="grey90"))
  p<-ggplot(df_long,aes(model_col,model_row,fill=value))+
    geom_tile(colour="white",linewidth=0.8)+geom_text(aes(label=label),size=3.2)+fs+
    labs(title=sprintf("DM Matrix: %s (tsCV-based)",type),
         subtitle=sprintf("Loss=%s|alt=%s",x$loss,x$alternative),x=NULL,y=NULL)+
    coord_fixed()+.bt()+theme(axis.text.x=element_text(angle=30,hjust=1),panel.grid=element_blank())
  print(p); invisible(p)
}


# ================================================================
# ██  PART 21 — compare_all_models()
# ================================================================

#' Compare all models using forecast::tsCV
#' @export
compare_all_models <- function(data,
                                rnn_models=c("lstm","gru","rnn"),
                                run_nbeats=FALSE,run_arima=TRUE,
                                run_arfima=TRUE,run_har=FALSE,run_garch=FALSE,
                                horizon=12L,rank_by="RMSE",rnn_strategy="recursive",
                                run_cv=TRUE,cv_holdout_frac=0.20,
                                cv_initial=NULL,cv_window=NULL,
                                rnn_config_obj=NULL,use_parallel=TRUE,
                                har_rv_col="value",garch_model="sGARCH",...){
  total_t0  <- .tic(); use_par <- use_parallel&&.is_parallel()
  all_names <- c(toupper(rnn_models%||%character(0)),
                 if(run_nbeats)"NBEATS",if(run_arima)"ARIMA",
                 if(run_arfima)"ARFIMA",if(run_har)"HAR",if(run_garch)"GARCH")
  cat(sprintf("\n── %s comparison | %d models | workers=%d | engine=%s\n",
    if(use_par)"Parallel" else "Sequential",length(all_names),.n_workers(),
    if(run_cv)"forecast::tsCV" else "hold-out"))

  train_model <- function(nm,dat){
    tryCatch(switch(nm,
      LSTM=auto_lstm(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                     cv_initial=cv_initial,cv_window=cv_window,config=rnn_config_obj,
                     use_parallel=FALSE,verbose=0L,...),
      GRU=auto_gru(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                    cv_initial=cv_initial,cv_window=cv_window,config=rnn_config_obj,
                    use_parallel=FALSE,verbose=0L,...),
      RNN=auto_rnn(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                    cv_initial=cv_initial,cv_window=cv_window,config=rnn_config_obj,
                    use_parallel=FALSE,verbose=0L,...),
      NBEATS=auto_nbeats(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                          cv_initial=cv_initial,cv_window=cv_window,config=rnn_config_obj,
                          use_parallel=FALSE,verbose=0L,...),
      ARIMA=auto_arima_ts(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                           cv_initial=cv_initial,cv_window=cv_window,verbose=0L),
      ARFIMA=auto_arfima_ts(dat,horizon=horizon,run_cv=run_cv,cv_holdout_frac=cv_holdout_frac,
                             cv_initial=cv_initial,cv_window=cv_window,verbose=0L),
      HAR=auto_har_rv(dat,rv_col=har_rv_col,horizon=min(horizon,5L),run_cv=run_cv,
                       cv_holdout_frac=cv_holdout_frac,cv_initial=cv_initial,cv_window=cv_window,verbose=0L),
      GARCH=auto_garch(dat,horizon=horizon,model=garch_model,run_cv=run_cv,
                        cv_holdout_frac=cv_holdout_frac,cv_initial=cv_initial,cv_window=cv_window,verbose=0L),
      stop(sprintf("Unknown model: %s",nm))
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

  .pull_m <- function(nm,res,metric){
    if(is.null(res)) return(NA_real_)
    if(inherits(res,"auto_rnn")){
      strat <- if(res$model_type=="nbeats")"nbeats" else rnn_strategy
      if(!is.null(res$cv_results)&&!is.null(res$cv_results$cv_metrics)){
        m <- res$cv_results$cv_metrics
        if("strategy"%in%names(m)){r<-m[m$strategy==strat,,drop=FALSE];if(nrow(r)>0) return(r[[metric]][1])}
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
  .pull_rt <- function(res){ rt<-res$runtime; if(is.null(rt)) NA_real_ else rt$total_sec%||%NA_real_ }

  metrics_show <- c("RMSE","MAE","MAPE","SMAPE")
  ranking <- dplyr::bind_rows(lapply(names(results),function(nm){
    row <- tibble(model=nm)
    for(m in metrics_show) row[[m]] <- .pull_m(nm,results[[nm]],m)
    row$runtime_sec <- .pull_rt(results[[nm]]); row$runtime_fmt <- .fmt_elapsed(row$runtime_sec); row
  }))%>%arrange(.data[[rank_by]])

  total_sec <- .toc(total_t0)
  cat("\n",.box(),"  Model Comparison\n",.box(),"\n",sep="")
  cat(sprintf("  Engine: %s | ranked by %s | total=%s\n",
    if(run_cv)"forecast::tsCV" else sprintf("hold-out %.0f%%",cv_holdout_frac*100),
    rank_by,.fmt_elapsed(total_sec)))
  cat(sprintf("  %-10s","Model"))
  for(m in metrics_show) cat(sprintf(" %10s",m))
  cat(sprintf(" %12s\n","Runtime"))
  cat(" ",strrep("-",64),"\n",sep="")
  for(i in seq_len(nrow(ranking))){
    cat(sprintf("  %-10s",ranking$model[i]))
    for(m in metrics_show) cat(sprintf(" %10s",.fmt(ranking[[m]][i],4)))
    cat(sprintf(" %12s",ranking$runtime_fmt[i]))
    if(i==1L) cat("  ← best"); cat("\n")
  }
  cat(.box(),"\n\n")

  model_pal <- unlist(.PAL[c("LSTM","GRU","RNN","NBEATS","ARIMA","ARFIMA","HAR","GARCH")])
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
    labs(title="Model Comparison — OOS Metrics (forecast::tsCV)",
         subtitle=sprintf("Ranked by %s | total=%s | %d workers",
           rank_by,.fmt_elapsed(total_sec),.n_workers()),x=NULL,y="Value")+bt
  print(comp_plot)

  rt_plot <- ranking%>%filter(!is.na(runtime_sec))%>%mutate(model=factor(model,levels=model))%>%
    ggplot(aes(model,runtime_sec,fill=model))+
    geom_col(width=0.55,show.legend=FALSE)+geom_text(aes(label=runtime_fmt),vjust=-0.4,size=3)+
    scale_fill_manual(values=model_pal,breaks=names(model_pal))+
    labs(title="Runtime Comparison",
         subtitle=sprintf("total=%s | workers=%d",.fmt_elapsed(total_sec),.n_workers()),
         x=NULL,y="Seconds")+bt
  print(rt_plot)

  list(ranking=ranking,results=results,plot=comp_plot,runtime_plot=rt_plot,total_sec=total_sec)
}


# ================================================================
# ██  PART 22 — BENCHMARK
# ================================================================

#' Benchmark tsCV parallel speedup
#' @export
benchmark_parallel <- function(data,n_folds=3L,horizon=6L,...){
  cat("\n── Parallel Speedup Benchmark (tsCV) ──────────────────────\n")
  cat(sprintf("   Available: %d | Current workers: %d\n",
              parallelly::availableCores(),.n_workers()))
  df <- data%>%rename(date=1,value=2)%>%arrange(date)%>%select(date,value)
  n  <- nrow(df); lags <- 12L
  hp <- list(hidden_size=32L,num_layers=1L,dropout=0.1,dense_units=16L,lr=1e-3,batch_size=16L)
  cv_init <- max(lags+horizon+10L,as.integer(n*0.5))
  results <- tibble(component=character(),sequential_sec=double(),
                    parallel_sec=double(),speedup=double(),workers=integer())

  cat("  Testing tsCV…\n")
  tf_reset_parallel()
  t_seq <- .tic()
  .tscv_deep(df,lags,horizon,hp,"lstm",cv_init,NULL,"recursive",
              20L,5L,0.15,0.5,NULL,1.0,TRUE,"cpu",0L,use_parallel=FALSE)
  seq_cv <- .toc(t_seq)
  tf_setup_parallel(verbose=FALSE)
  t_par <- .tic()
  .tscv_deep(df,lags,horizon,hp,"lstm",cv_init,NULL,"recursive",
              20L,5L,0.15,0.5,NULL,1.0,TRUE,"cpu",0L,use_parallel=TRUE)
  par_cv <- .toc(t_par)
  tf_reset_parallel()
  results <- rbind(results,tibble(component="tsCV",sequential_sec=seq_cv,parallel_sec=par_cv,
    speedup=seq_cv/max(par_cv,0.01),workers=.n_workers()))
  cat("\n── Speedup Results ─────────────────────────────────────────\n")
  print(results%>%mutate(across(c(sequential_sec,parallel_sec),~round(.,2)),speedup=round(speedup,2)))
  cat(sprintf("\n   Avg speedup: %.2fx\n",mean(results$speedup,na.rm=TRUE)))
  invisible(results)
}


# ================================================================
# ██  PART 23 — EXAMPLE USAGE
# ================================================================

# ── Uncomment to run ─────────────────────────────────────────────────
#
# source("torchforecast.R")
# library(zoo)
#
# df <- data.frame(
#   date  = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
#   value = as.numeric(AirPassengers)
# )
# df_fin <- data.frame(
#   date  = seq(as.Date("2020-01-01"), by = "day", length.out = 500),
#   value = cumsum(rnorm(500, 0, 0.01))
# )
#
# # ── 1. Setup parallel backend ─────────────────────────────────────
# tf_setup_parallel()                     # auto-detect cores
# tf_setup_parallel(workers = 4L)         # fixed 4 workers
# tf_reset_parallel()                     # sequential mode
#
# # ── 2. Configuration ─────────────────────────────────────────────
# cfg <- rnn_config(
#   run_bo = TRUE, bo_iter = 12L,
#   bo_bounds = list(hidden_size = c(32L, 128L)),
#   run_cv = TRUE,                 # uses forecast::tsCV
#   cv_window = NULL,              # NULL = expanding window
#   # cv_window = 84L,             # fixed sliding window of 84 obs
#   seed = 42L
# )
#
# # ── 3. Individual models ──────────────────────────────────────────
# lstm_res   <- auto_lstm(df, horizon = 12L, config = cfg)
# gru_res    <- auto_gru(df,  horizon = 12L, config = cfg)
# nbeats_res <- auto_nbeats(df, horizon = 12L)
# arima_res  <- auto_arima_ts(df, horizon = 12L, run_cv = TRUE, cv_window = NULL)
# arfima_res <- auto_arfima_ts(df, horizon = 12L)
# har_res    <- auto_har_rv(df_fin, horizon = 5L, variant = "harj")
# garch_res  <- auto_garch(df_fin, horizon = 10L, model = "gjrGARCH",
#                           run_cv = TRUE, cv_window = 252L)  # 1-year sliding window
#
# # ── 4. Inspect ───────────────────────────────────────────────────
# print(lstm_res)                  # shows tsCV overall metrics
# cv_performance(lstm_res)         # by-horizon breakdown + overall
# plot(lstm_res)                   # includes tsCV by-horizon panel
# plot_oos_fit(lstm_res)           # 6-panel OOS actual vs fitted
# plot_oos_fit(arima_res)
#
# # ── 5. DM test (from tsCV error vectors) ─────────────────────────
# dm_out <- dm_compare(lstm_res, arima_res)
# print(dm_out$dm); print(dm_out$dm_horizon)
# plot(dm_out$dm)
#
# # Pairwise matrix (parallel)
# tf_setup_parallel()
# oos_all <- list(LSTM=.get_oos(lstm_res), GRU=.get_oos(gru_res), ARIMA=.get_oos(arima_res))
# dm_mat  <- dm_matrix(oos_all, loss = "MSE", use_parallel = TRUE)
# print(dm_mat); plot(dm_mat, type = "pvalue")
#
# # ── 6. Full comparison ────────────────────────────────────────────
# comp <- compare_all_models(
#   data       = df,
#   rnn_models = c("lstm","gru","rnn"),
#   run_nbeats = TRUE, run_arima = TRUE, run_arfima = TRUE,
#   horizon    = 12L, rank_by   = "RMSE",
#   run_cv     = TRUE, cv_window = NULL,      # expanding tsCV
#   rnn_config_obj = rnn_config(bo_iter = 8L),
#   use_parallel   = TRUE
# )
# print(comp$ranking)
#
# # ── 7. Sliding window tsCV ────────────────────────────────────────
# # cv_window = integer means fixed-size window (not expanding)
# lstm_sw <- auto_lstm(df, horizon = 12L,
#                      run_cv = TRUE, cv_window = 84L,
#                      cv_initial = 84L)
#
# # ── 8. Hold-out mode ─────────────────────────────────────────────
# lstm_ho <- auto_lstm(df, horizon = 12L, run_cv = FALSE, cv_holdout_frac = 0.20)
# plot_oos_fit(lstm_ho)   # fitted vs TRUE test-set values
#
# # ── 9. Benchmark ─────────────────────────────────────────────────
# bench <- benchmark_parallel(df)
#
# # ── 10. Reset ────────────────────────────────────────────────────
# tf_reset_parallel()
