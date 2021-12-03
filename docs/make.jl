using NRST
using Documenter

DocMeta.setdocmeta!(NRST, :DocTestSetup, :(using NRST); recursive=true)

makedocs(;
    modules=[NRST],
    authors="Miguel Biron <miguelbl9@gmail.com> and contributors",
    repo="https://github.com/miguel.biron/NRST.jl/blob/{commit}{path}#{line}",
    sitename="NRST.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://miguel.biron.github.io/NRST.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/miguel.biron/NRST.jl",
    devbranch="main",
)
