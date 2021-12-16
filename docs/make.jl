using Documenter, NRST

makedocs(
    sitename = "NRST.jl",
    modules  = [NRST],
    pages = [
        "Home" => "index.md",
        "Case studies" => [
            "Toy Gaussians" => "toy_Gaussians.md"
        ]
    ]
)
