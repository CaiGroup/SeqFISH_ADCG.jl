using Documenter
using SeqFISH_ADCG

makedocs(
    sitename = "SeqFISH_ADCG",
    format = Documenter.HTML(),
    modules = [SeqFISH_ADCG]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
