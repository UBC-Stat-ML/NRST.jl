using Documenter
using DemoCards
using NRST
using LiveServer: serve

# only way to deal with annoying Literate.jl call to git
Sys.which(::String)=nothing

# make DemoCards
examples, examples_cb, examples_assets = makedemos(
    "examples", edit_branch = "main"
)

# if there are generated css assets, pass it to Documenter.HTML
assets = []
isnothing(examples_assets) || (push!(assets, examples_assets))

format = Documenter.HTML(assets = assets)
makedocs(
    format = format,
    sitename = "NRST.jl",
    modules  = [NRST],
    pages = [
        "Home" => "index.md",
        examples,
        "Internals" => "internals.md",
    ]
)

# postprocess after makedocs
examples_cb()

# serve the docs locally
serve(dir="./docs/build", launch_browser=true)
