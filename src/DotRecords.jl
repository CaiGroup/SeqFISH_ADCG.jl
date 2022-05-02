using NearestNeighbors
using DataFrames

mutable struct DotRecords
    records :: DataFrame
    last_iteration :: DataFrame
    last_iteration_tree :: KDTree
end
