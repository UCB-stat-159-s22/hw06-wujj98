.PHONY: env
env :
	mamba env create -f environment.yml --name ligo
	conda activate ligo
	python -m ipykernel install --user --name ligo --display-name "HW06 LIGO"

.PHONY: html
html :
	jupyter-book build .
    
.PHONY: html-hub
html-hub : 
	jupyter-book config sphinx .
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}/proxy/absolute/8000
   
.PHONY: clean
clean : 
	rm -f figures/*.png
	rm -f audio/*.wav
	rm -r _build/*
  
.PHONY: all
all : 
	jupyter execute index.ipynb --kernel_name=ligo