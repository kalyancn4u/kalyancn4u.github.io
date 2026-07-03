# # frozen_string_literal: true

# require "json"

# module SearchIndex
#   class SearchPage < Jekyll::Page
#     def initialize(site, json)
#       @site = site
#       @base = site.source
#       @dir  = ""
#       @name = "search.json"

#       process(@name)

#       self.data = {
#         "layout" => nil,
#         "sitemap" => false
#       }

#       @content = json
#     end

#     def render(*)
#       # Do nothing; content is already JSON
#     end

#     def output
#       @content
#     end
#   end

#   class Generator < Jekyll::Generator
#     safe true
#     priority :lowest

#     def generate(site)
#       items = []

#       docs = []

#       docs.concat(site.posts.docs) if site.respond_to?(:posts)

#       docs.concat(site.pages)

#       site.collections.each do |name, collection|
#         next if name == "posts"

#         docs.concat(collection.docs)
#       end

#       docs.each do |doc|
#         next unless doc.output_ext == ".html"
#         next if doc.data["search"] == false
#         next if doc.url.nil?

#         content =
#           doc.content
#              .gsub(/<[^>]+>/, " ")
#              .gsub(/\{\{.*?\}\}/m, " ")
#              .gsub(/\{%.*?%\}/m, " ")
#              .gsub(/\s+/, " ")
#              .strip

#         items << {
#           title: doc.data["title"] || "",
#           url: doc.url,
#           content: content,
#           aliases: Array(doc.data["aliases"]),
#           keywords: Array(doc.data["keywords"]),
#           shortcuts: Array(doc.data["shortcuts"]),
#           tags: Array(doc.data["tags"]),
#           categories: Array(doc.data["categories"])
#         }
#       end

#       json = JSON.pretty_generate(items)

#       site.pages << SearchPage.new(site, json)

#       Jekyll.logger.info(
#         "Search",
#         "Generated search.json (#{items.size} documents)"
#       )
#     end
#   end
# end
