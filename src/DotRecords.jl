using NearestNeighbors
using DataFrames
export get_mw_dots

mutable struct DotRecords
    records :: DataFrame
    last_iteration :: DataFrame
    last_iteration_tree :: KDTree
end

"""
    get_mw_dots(records :: DataFrame, w :: Real)

Look up the ADCG result at a minimum weight larger than the final minimum weight given for an ADCG run.

Returns a dataframe of results.
"""
function get_mw_dots(records :: DataFrame, w :: Real)
    results = filter(d -> d.lowest_mw <= w && d.highest_mw >= w, records)
    results = results[:, 1:(ncol(results)-2)]
    return results
end