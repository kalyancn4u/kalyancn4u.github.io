console.log("✓ search.js loaded");

document.addEventListener("DOMContentLoaded", async () => {

    console.log("✓ DOM loaded");

    const input = document.getElementById("search-input");
    const results = document.getElementById("search-results");

    // if (!input || !results) return;

    if (!input || !results) {
        console.error("Search elements not found");
        return;
    }

    console.log("✓ Elements found");

    // const response = await fetch("/assets/js/data/search.json");
    const response = await fetch("/search.json");
    console.log("✓ Fetch completed");

    const documents = await response.json();
    console.log(`✓ Loaded ${documents.length} documents`);

    const fuse = new Fuse(documents, {
        includeScore: true,
        includeMatches: true,
        shouldSort: true,
        threshold: 0.30,
        ignoreLocation: true,
        findAllMatches: true,
        minMatchCharLength: 2,

        keys: [
            { name: "title", weight: 10 },
            { name: "aliases", weight: 9 },
            { name: "shortcuts", weight: 8 },
            { name: "keywords", weight: 7 },
            { name: "tags", weight: 6 },
            { name: "categories", weight: 5 },
            { name: "content", weight: 1 }
        ]
    });

    console.log("✓ Fuse created");

    input.addEventListener("input", () => {

        const query = input.value.trim();
        console.log("Typed:", query);

        results.innerHTML = "";
        if (query.length < 2) return;

        const matches = fuse.search(query);
        console.log(matches);

        matches.slice(0, 20).forEach(({ item }) => {

            const div = document.createElement("div");
            div.className = "search-result";

            div.innerHTML = `
                <h4>
                    <a href="${item.url}">
                        ${item.title}
                    </a>
                </h4>

                <p>${(item.content || "").substring(0,180)}...</p>
            `;

            results.appendChild(div);

        });

    });

});
