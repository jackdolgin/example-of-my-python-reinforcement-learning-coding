# copied and modified from spacyr spacy_install.R, https://github.com/quanteda/spacyr/blob/342acf8167c4177b364dd84c3195290ff226b9a9/R/spacy_install.R

source(here("Analysis", "py_scripts", "venv_setup", "utils.R"))


conda_args <- reticulate:::conda_args

# need to rename `spacy_install` to a project-specific title, and same for `envnam`
# for that matter, need to replace the word `spacy` any time it appears in this script, be it comments or code

spacy_install <- function(conda = "auto",
                          python_version = "3.7",
                          envname = "spacy_condaenv",
                          prompt = TRUE) {
  # verify os
  if (!is_windows() && !is_osx() && !is_linux()) {
    stop("This function is available only for Windows, Mac, and Linux")
  }
  
  # resolve and look for conda
  conda <- tryCatch(reticulate::conda_binary(conda), error = function(e) NULL)
  have_conda <- !is.null(conda)
  
  # mac and linux
  if (is_unix()) {
    
      # check for explicit conda method
      
      # validate that we have conda
      if (!have_conda) {
        cat("No conda was found in the system. ")
        ans <- utils::menu(c("No", "Yes"), title = "Do you want us to download miniconda in ~/miniconda?")
        if (ans == 2) {
          install_miniconda()
          conda <- tryCatch(reticulate::conda_binary("auto"), error = function(e) NULL)
        } else stop("Conda environment installation failed (no conda binary found)\n", call. = FALSE)
      }
      
      # process the installation of spacy
      process_spacy_installation_conda(conda, python_version,  envname, prompt = prompt)
    
    # windows installation
  } else {
    
    # determine whether we have system python
    python_versions <- reticulate::py_versions_windows()
    python_versions <- python_versions[python_versions$type == "PythonCore", ]
    python_versions <- python_versions[python_versions$version %in% c("3.5", "3.6"), ]
    python_versions <- python_versions[python_versions$arch == "x64", ]
    have_system <- nrow(python_versions) > 0
    if (have_system)
      python_system_version <- python_versions[1, ]
    
    # validate that we have conda
    if (!have_conda) {
      stop("Conda installation failed (no conda binary found)\n\n",
           "Install Anaconda 3.x for Windows (https://www.anaconda.com/download/#windows)\n",
           "before installing spaCy",
           call. = FALSE)
    }
    
    # process the installation of spacy
    process_spacy_installation_conda(conda, python_version, envname, prompt)
    
  }
  message("\nInstallation complete.\n")
  
  invisible(NULL)
}

process_spacy_installation_conda <- function(conda, python_version, prompt,
                                             envname) {
  
  conda_envs <- reticulate::conda_list(conda = conda)
  if (prompt) {
    ans <- utils::menu(c("No", "Yes"), title = "Proceed?")
    if (ans == 1) stop("condaenv setup is cancelled by user", call. = FALSE)
  }
  conda_env <- subset(conda_envs, conda_envs$name == envname)
  if (nrow(conda_env) == 1) {
    cat("Using existing conda environment ", envname, " for installation.")
    python <- conda_env$python
  }
  else {
    cat("A new conda environment", paste0('"', envname, '"'), "will be created")
    cat("Creating", envname, "conda environment for installation...")
    python_version_reticulate <- ifelse(is.null(python_version), "python=3.7",
                              sprintf("python=%s", python_version))
    python <- reticulate::conda_create(envname,
                                       packages = python_version_reticulate,
                                       conda = conda)
  }
  
  py_packages <- here("Analysis", "py_scripts", "venv_setup",
                      "requirements.txt") %>%
    scan(what="", sep="\n") %>%
    str_extract("[:alpha:]*")
  
  reticulate::conda_install(envname, packages = py_packages, conda = conda)
  
}

install_miniconda <- function() {
  if (is_osx()) {
    message("Downloading installation script")
    system(paste(
      "curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda.sh;",
      "echo \"Running installation script\";",
      "bash ~/miniconda.sh -b -p $HOME/miniconda"))
    system('echo \'export PATH="$PATH:$HOME/miniconda/bin"\' >> $HOME/.bash_profile; rm ~/miniconda.sh')
    message("Installation of miniconda complete")
  } else if (is_linux()) {
    message("Downloading installation script")
    system(paste(
      "wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh;",
      "echo \"Running installation script\";",
      "bash ~/miniconda.sh -b -p $HOME/miniconda"))
    system('echo \'export PATH="$PATH:$HOME/miniconda/bin"\' >> $HOME/.bashrc; rm ~/miniconda.sh')
    message("Installation of miniconda complete")
  } else {
    stop("miniconda installation is available only for Mac or Linux")
  }
}
