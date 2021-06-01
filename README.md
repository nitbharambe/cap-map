# cap-map
Grid capacity map generation using pandapower.

Use cap_comb to get the map. First half of the file (till line 47) is a time intesive is calculation of power flow by pfa.all_cap_map function. The result is then stored in sampdata directory. The next section visualisation plots that map. So running the 2 sections separately is advisable

Two features presented in the simple model are visible in this map.
1) Maximum capacity possible at each bus is mentioned by the tooltip on nodes.
2) Analysis of a single power flow results in terms of loading and voltage limits shown like a heat map 

Mapbox scatter function of viz.py to be used for plotting geodata.
Included another method of calculating powerflow by joining profiles directly. Not used in the final file but saved functions.
