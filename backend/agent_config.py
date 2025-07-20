
from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage, add_messages, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools.google_search import GoogleSearchResults
from langchain_core.tools import tool
import yfinance as yf
import pandas as pd
from logger import logger
import matplotlib.pyplot as plt
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import os
import io

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# -- LLM and Tools Setup --
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Google Search Tool
search_api_wrapper = GoogleSearchAPIWrapper()
search_tool = GoogleSearchResults(api_wrapper=search_api_wrapper)


# tools
@tool(description="""
    Fetch stock price data for one or multiple tickers with options for period or date range.
    
    Parameters:
    - tickers (list of str): List of ticker symbols (eg. 'AAPL', 'MSFT', ...)
    - period (str, optional): Data period (e.g., '1d', '5d', '1mo', '3mo', '1y', '5y', 'max').
                              Used if start and end are not specified.
    - start (str or datetime, optional): Start date in 'YYYY-MM-DD' format or datetime object.
    - end (str or datetime, optional): End date in 'YYYY-MM-DD' format or datetime object.
    
    Returns:
    - A dictionary with tickers as keys (strings) and strings as values (representing a CSV-file, historical price data).
      
      
    Note: If start and end are given, period is ignored.
    """)
def fetch_stock_data(tickers: list[str], period: Optional[str]=None, start: Optional[str]=None, end: Optional[str]=None):
    logger.info(f"""called fetch_stock_data with tickers: {tickers}, period: {period},
                start: {start}, end: {end}""", extra={'component': 'NODE'})

    data = {}
    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        if start and end:
            data[ticker] = ticker_obj.history(start=start, end=end).drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
        else:
            data[ticker] = ticker_obj.history(period=period).drop(['Volume', 'Dividends', 'Stock Splits'], axis=1)
    return {
        "content": {ticker: df.to_csv(index=False) for ticker, df in data.items()},
        "type": "dict",
        "response": f"Fetched stock data for tickers {tickers}"
    }


@tool(description="""
Tool for plotting out stock prices and saving the image as a .png.
Inputs: timestamps (list of str in YYYY-MM-DD format, eg. ['2024-01-01', '2024-01-02', ...]),
close_prices (list of floats, eg. [200.0, 200.1, ...]), tickers (list of str, eg. ['AAPL', 'GOOGLE', ...]),
filename (string), name of the image to be saved.""")
def plot_prices(timestamps: list[str], close_prices: list[float], tickers: list[str], filename: str):
    logger.info(f"called plot_prices with timestamps: {timestamps}, close_prices: {close_prices}, tickers: {tickers}",
                extra={'component': 'NODE'})
    plt.figure(figsize=(10,6))
    for ticker in tickers:
        plt.plot(timestamps, close_prices, label=ticker)

    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"frontend/generated_content/{filename}")
    return {
        "response": f"Saved {filename} successfully.",
        "content": None,
        "type": "image"
    }

@tool(description="""Tool for saving a CSV string (eg. 'Open,High,Low,Close\r\n503.04998779296875,...)
to a .csv file. Inputs: data: the CSV string (str), filename: the filename (str).""")
def save_csv(data: str, filename: str):
    logger.info(f"called save_csv with data: {data}, filename: {filename}", extra={'component': 'NODE'})
    pd.read_csv(io.StringIO(data)).to_csv("frontend/generated_content/" + filename)
    return {
        "response": f'Saved the CSV content to "frontend/generated_content/{filename}".',
        "type": "csv-file",
        "content": f"frontend/generated_content/{filename}"
    }

@tool(description="""Tool for reading the contents of the csv file.
Inputs: filename (string), x_data_name (the name of the wanted x_data column, string),
y_data_name (the name of the wanted y_data column, string).
Output: the x_data and y_data, both as lists, wrapped inside of a list containing these two items.""")
def read_csv(filename: str, x_data_name: str, y_data_name: str):
    logger.info(f"called read_csv with filename: {filename}, x_data_name: {x_data_name}, y_data_name: {y_data_name}", extra={'component': 'NODE'})
    df = pd.read_csv(filename)
    if x_data_name not in df.columns:
        logger.error(f"{x_data_name} not in the columns of the df!")
        return f"{x_data_name} not in the columns of the dataframe, please add this first."
    if y_data_name not in df.columns:
        logger.error(f"{y_data_name} not in the columns of the df!")
        return f"{y_data_name} not in the columns of the dataframe, please add this first."
    xdata = df[x_data_name].to_list()
    ydata = df[y_data_name].to_list()
    return {
        "content": [xdata, ydata],
        "response": "gotten the x- and y-data of the csv-file",
        "type": "list of lists"
    }

@tool(description="""Tool for adding a column to a dataframe stored in a csv-file. Inputs:
filename (string), rows (list of strings, the rows to add).""")
def add_rows(filename: str, row_names: list[str], row_contents: list[list]):
    df = pd.read_csv(filename)
    if len(row_names) != len(row_contents):
        logger.error(f"In add_rows: Row names and Row contents must have the same length!")
        return "Row names and Row contents must have the same length!"
    for i in range(row_names):
        df[row_names[i]] = row_contents[i]
    df.to_csv(filename)
    return {
        "response": "Added the rows to the csv-file",
        "type": "modification",
        "content": None
    }

tools = [search_tool, fetch_stock_data, plot_prices, save_csv, read_csv, add_rows]
tools_by_name = {tool.name: tool for tool in tools}
model = llm.bind_tools(tools)



def call_model(state: AgentState, config: RunnableConfig = None) -> AgentState:
    logger.info('called call_model', extra={'component': 'NODE'})
    return {"messages": [model.invoke(state["messages"], config)]}
    
def call_tool(state: AgentState):
    logger.info('called call_tool', extra={'component': 'NODE'})
    last_msg = state["messages"][-1]
    for tool_call in getattr(last_msg, "tool_calls", None):
        logger.info(f"Invoking tool: {tool_call['name']}, args: {tool_call['args']}, id: {tool_call['id']}",
                     extra={'component': 'NODE'})

    # Check if last message has tool_calls attribute and it is not empty
    tool_calls = getattr(last_msg, "tool_calls", None)
    if not tool_calls:
        # No tool calls to process, return empty message list or handle appropriately
        return {"messages": []}

    # Process all tool calls
    return {
        "messages": [
            ToolMessage(
                content=tools_by_name[tool_call["name"]].invoke(tool_call["args"]),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for tool_call in tool_calls
        ]
    }


def should_continue(state: AgentState) -> str:
    logger.info('called should_continue', extra={'component': 'NODE'})
    if hasattr(state["messages"][-1], "tool_calls"):
        return "continue" if state["messages"][-1].tool_calls else "end"
    return "continue"

def final_output(state: AgentState, config: RunnableConfig = None) -> str:
    msg = """
    Using the results from previous tool executions,
    explain to the user what was accomplished and summarize the findings in a conversational,
    accessible manner, and end the session after having achieved this.
        """
    messages = state["messages"] + [SystemMessage(content=msg)]
    result = model.invoke(messages, config)
    return {"messages:", [result]}



workflow = StateGraph(AgentState)
workflow.add_node("final", final_output)
workflow.add_node("llm", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": "final"})
workflow.add_edge("final", END)
workflow.add_edge("tools", "llm")
graph = workflow.compile()