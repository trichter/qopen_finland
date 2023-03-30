# overwrite part of the configuration with command line options
# variables are defined for better overview
CONFGO='{"skip": {"coda_window": 5, "num_pairs": 5}}'
CONFGO_LESS_FREQS='{"skip": {"coda_window": 5, "num_pairs": 5}, "freqs": {"width": 1, "cfreqs": [6.0, 12.0, 24.0]}}'
EVENTS2020="../data/events2020_ML>=0.csv"
CONF2020='{"use_picks": false, "bulk_window": null, "coda_window": ["S+0s", ["OT+18s", "2SNR"]], "skip": {"coda_window": 2, "num_pairs": 3}, "plot_sds_options": {"fname": "plots/sds.pdf", "nx": 8, "figsize": [14, 5], "annotate": true}}'
EVID2018=2018171232614IMS000000
EVID2020=2020105023127ISUHX00000
ALIGN_SITES="--align-sites --align-sites-value 0.25 --align-sites-station HE.MALM,HE.RUSK"
ALIGN_SITES_SURFACE="--align-sites --align-sites-station HE.HEL1,HE.HEL2,HE.HEL3,HE.HEL4,HE.HEL5,HE.NUR"
SMO='{"fc": null, "n": 1.74, "gamma": 2, "fc_lim": [5, 100], "num_points": 5}'

# initial tests
qopen go -e $EVID2018 --prefix "000_test/"
qopen go -e $EVID2020 --prefix "000_test_2020/" --events "$EVENTS2020" --overwrite-conf "$CONF2020"

# Qopen for 2018 dataset
qopen go --filter-events '["magnitude >= 1"]' --prefix 01_go/ --overwrite-conf "$CONFGO"
qopen fixed --filter-events '["magnitude >= 1"]' --input 01_go/results.json --prefix 02_sites/ $ALIGN_SITES --no-plot-results
qopen source --input 01_go/results.json --input-sites 02_sites/results.json --prefix 03_source/ --no-plot-results
qopen recalc_source --input 03_source/results.json --prefix 04_source_nconst/ --seismic-moment-options "$SMO"

# Qopen in monitoring mode for 2020 data
qopen source --input 01_go/results.json --input-sites 02_sites/results.json --prefix 06_source_2020/ --events $EVENTS2020 --overwrite-conf "$CONF2020" --no-plot-results
qopen recalc_source --input 06_source_2020/results.json --prefix 07_source_2020_nconst/ --seismic-moment-options "$SMO" --overwrite-conf "$CONF2020"

# dump fit data for later plotting
qopen go -e $EVID2018  --dump-fitpkl 01_go/fits_%s.pkl --overwrite-conf "$CONFGO_LESS_FREQS" --no-plots --output null
qopen source -e $EVID2020 --dump-fitpkl 06_source_2020/fits_%s.pkl --input 01_go/results.json --input-sites 02_sites/results.json --events $EVENTS2020 --overwrite-conf "$CONF2020" --no-plots --output null

# calculate site responses for all stations
qopen fixed --inventory ../data/stations2018.xml --filter-events '["magnitude >= 1"]' --input 01_go/results.json  --prefix 05_sites_all_stations/ $ALIGN_SITES --no-plot-results
