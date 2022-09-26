push!(LOAD_PATH,"../src/")
using Documenter
using SeqFISH_ADCG

_PAGES = [
    "index.md",
    "installation.md",
    "example_FitDots.md",
    "api_reference.md"
]

makedocs(
    sitename = "SeqFISH_ADCG",
    format = Documenter.HTML(prettyurls=false),
    modules = [SeqFISH_ADCG],
    pages = _PAGES
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
