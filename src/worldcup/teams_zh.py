"""FIFA 3-letter code → Chinese team name (WC2026 48 teams + common extras).

Shared by the digest (display) and the scout layer (injected into LLM prompts so
the model never confuses e.g. COD 刚果(金) with CIV 科特迪瓦).
"""

CODE_ZH: dict[str, str] = {
    "ARG": "阿根廷", "ALG": "阿尔及利亚", "AUS": "澳大利亚", "AUT": "奥地利",
    "BEL": "比利时", "BIH": "波黑", "BRA": "巴西", "CAN": "加拿大", "CIV": "科特迪瓦",
    "COD": "刚果(金)", "COL": "哥伦比亚", "CPV": "佛得角", "CRO": "克罗地亚",
    "CUW": "库拉索", "CZE": "捷克", "ECU": "厄瓜多尔", "EGY": "埃及", "ENG": "英格兰",
    "ESP": "西班牙", "FRA": "法国", "GER": "德国", "GHA": "加纳", "HAI": "海地",
    "IRN": "伊朗", "IRQ": "伊拉克", "JOR": "约旦", "JPN": "日本", "KOR": "韩国",
    "KSA": "沙特", "MAR": "摩洛哥", "MEX": "墨西哥", "NED": "荷兰", "NOR": "挪威",
    "NZL": "新西兰", "PAN": "巴拿马", "PAR": "巴拉圭", "POR": "葡萄牙", "QAT": "卡塔尔",
    "RSA": "南非", "SCO": "苏格兰", "SEN": "塞内加尔", "SUI": "瑞士", "SWE": "瑞典",
    "TUN": "突尼斯", "TUR": "土耳其", "URU": "乌拉圭", "USA": "美国", "UZB": "乌兹别克斯坦",
    "ITA": "意大利", "NGA": "尼日利亚", "CHN": "中国", "WAL": "威尔士", "SRB": "塞尔维亚",
}
