from serpapi import GoogleSearch
from .base_tool import BaseTool
import os
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

class SearchGoogle(BaseTool):
    name = "SearchGoogle"
    # api = "SearchGoogle(query: str)"
    description_en = "Search Google for the given query."
    description_zh = "搜索给定的查询."

    def call(self, query: str):
        search = GoogleSearch({
            "q": query,
            "api_key": SERPAPI_API_KEY
        })
        results = search.get_dict()
        return self._process_response(results)

    @staticmethod
    def _process_response(res: dict) -> str:
        """Process response from SerpAPI.
        from langchain_community/utilities/serpapi.py
        """
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box_list" in res.keys():
            res["answer_box"] = res["answer_box_list"]
        if "answer_box" in res.keys():
            answer_box = res["answer_box"]
            if isinstance(answer_box, list):
                answer_box = answer_box[0]
            if "result" in answer_box.keys():
                return answer_box["result"]
            elif "answer" in answer_box.keys():
                return answer_box["answer"]
            elif "snippet" in answer_box.keys():
                return answer_box["snippet"]
            elif "snippet_highlighted_words" in answer_box.keys():
                return answer_box["snippet_highlighted_words"]
            else:
                answer = {}
                for key, value in answer_box.items():
                    if not isinstance(value, (list, dict)) and not (
                        isinstance(value, str) and value.startswith("http")
                    ):
                        answer[key] = value
                return str(answer)
        elif "events_results" in res.keys():
            return res["events_results"][:10]
        elif "sports_results" in res.keys():
            return res["sports_results"]
        elif "top_stories" in res.keys():
            return res["top_stories"]
        elif "news_results" in res.keys():
            return res["news_results"]
        elif "jobs_results" in res.keys() and "jobs" in res["jobs_results"].keys():
            return res["jobs_results"]["jobs"]
        elif (
            "shopping_results" in res.keys()
            and "title" in res["shopping_results"][0].keys()
        ):
            return res["shopping_results"][:3]
        elif "questions_and_answers" in res.keys():
            return res["questions_and_answers"]
        elif (
            "popular_destinations" in res.keys()
            and "destinations" in res["popular_destinations"].keys()
        ):
            return res["popular_destinations"]["destinations"]
        elif "top_sights" in res.keys() and "sights" in res["top_sights"].keys():
            return res["top_sights"]["sights"]
        elif (
            "images_results" in res.keys()
            and "thumbnail" in res["images_results"][0].keys()
        ):
            return str([item["thumbnail"] for item in res["images_results"][:10]])

        snippets = []
        if "knowledge_graph" in res.keys():
            knowledge_graph = res["knowledge_graph"]
            title = knowledge_graph["title"] if "title" in knowledge_graph else ""
            if "description" in knowledge_graph.keys():
                snippets.append(knowledge_graph["description"])
            for key, value in knowledge_graph.items():
                if (
                    isinstance(key, str)
                    and isinstance(value, str)
                    and key not in ["title", "description"]
                    and not key.endswith("_stick")
                    and not key.endswith("_link")
                    and not value.startswith("http")
                ):
                    snippets.append(f"{title} {key}: {value}.")

        for organic_result in res.get("organic_results", []):
            if "snippet" in organic_result.keys():
                snippets.append(organic_result["snippet"])
            elif "snippet_highlighted_words" in organic_result.keys():
                snippets.append(organic_result["snippet_highlighted_words"])
            elif "rich_snippet" in organic_result.keys():
                snippets.append(organic_result["rich_snippet"])
            elif "rich_snippet_table" in organic_result.keys():
                snippets.append(organic_result["rich_snippet_table"])
            elif "link" in organic_result.keys():
                snippets.append(organic_result["link"])

        if "buying_guide" in res.keys():
            snippets.append(res["buying_guide"])
        if "local_results" in res and isinstance(res["local_results"], list):
            snippets += res["local_results"]
        if (
            "local_results" in res.keys()
            and isinstance(res["local_results"], dict)
            and "places" in res["local_results"].keys()
        ):
            snippets.append(res["local_results"]["places"])
        if len(snippets) > 0:
            return str(snippets)
        else:
            return "No good search result found"


if __name__ == '__main__':
    tool = SearchGoogle()
    print(tool.call("Python programming language"))