document.addEventListener("DOMContentLoaded", async () => {

  const input = document.getElementById("search-input");
  const results = document.getElementById("search-results");

  if (!input || !results) return;

  # const response = await fetch("/search.json");
  const response = await fetch("/assets/js/data/search.json");
  const documents = await response.json();
  
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

  input.addEventListener("input", function () {

    const query = input.value.trim();

    results.innerHTML = "";

    if (query.length < 2) return;

    const matches = fuse.search(query);

    matches.slice(0, 20).forEach(result => {

      const doc = result.item;

      const div = document.createElement("div");

      div.className = "search-result";

      div.innerHTML = `
        <h4>
          <a href="${doc.url}">
            ${doc.title}
          </a>
        </h4>

        <p>${(doc.excerpt || doc.content || "")
          .substring(0,180)}
        ...</p>
      `;

      results.appendChild(div);

    });

  });

});
