# data branch: run history of the fork

`run-history.csv` has one row per measured run (append only, rows are never edited): build, commit, model, test, t/s, prompt t/s,
hit rate, machine, power limits (`pl`), campaign and the settings used. It is written by `fork/run.py` and published here by
`publish_history.py` so the numbers behind the README can be checked. Code lives on branch `main`; this branch holds data only.

Columns: ts,commit,bno,tps,pp,hit,model,args,test,hw,ok,build,md5,campaign,note,ppl,spp,n_gen,origin,env,cmd,pl,proto,rep,seed,branch,patch,exp,arm,ready_s,wall_s,rss_kb,major_faults,uploads,evictions,up_mib,ddr_gbs,pcie_gbs,ddr_util,pcie_util,cpu_util,gpu_util,placement,rungs,held_gib,mem_limit,cache_state,state,sys
