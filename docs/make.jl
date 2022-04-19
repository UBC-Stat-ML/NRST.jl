using Documenter
using DemoCards
using NRST

examples, examples_cb, examples_assets = makedemos("examples")

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

# 3. postprocess after makedocs
examples_cb()
