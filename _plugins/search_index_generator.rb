# frozen_string_literal: true

require "json"

module SearchIndex
  class Generator < Jekyll::Generator
    safe true
    priority :lowest

    def generate(site)
      items = []

      docs = []
      docs.concat(site.posts.docs) if site.respond_to?(:posts)
      docs.concat(site.pages)
      docs.concat(site.collections["tabs"].docs) if site.collections["tabs"]

      docs.each do |doc|
        next unless doc.output_ext == ".html"

        items << {
          title: doc.data["title"] || "",
          url: doc.url,
          content: doc.content
                     .gsub(/<[^>]*>/, " ")
                     .gsub(/\s+/, " ")
                     .strip,
          aliases: Array(doc.data["aliases"])
        }
      end

      File.write(
        File.join(site.dest, "search.json"),
        JSON.pretty_generate(items)
      )

      Jekyll.logger.info "Search:", "Generated search.json (#{items.size} pages)"
    end
  end
end
