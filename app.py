
from pathlib import Path
from urllib.parse import quote_plus

import cloudpickle
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

ENGINE_PATH = BASE_DIR / "recommendation_engine.pkl"
INVENTORY_PATH = BASE_DIR / "fridge_inventory.pkl"
FEEDBACK_PATH = BASE_DIR / "feedback_log.pkl"


st.set_page_config(
    page_title="냉장고 레시피 추천",
    page_icon="🍳",
    layout="wide"
)


@st.cache_resource
def load_engine():
    with open(ENGINE_PATH, "rb") as file:
        return cloudpickle.load(file)


def load_inventory():
    return pd.read_pickle(INVENTORY_PATH)


engine = load_engine()

if "fridge_df" not in st.session_state:
    st.session_state.fridge_df = load_inventory()


# 기존 냉장고 데이터에 재료 분류 추가
if "ingredient_group" not in st.session_state.fridge_df.columns:
    st.session_state.fridge_df["ingredient_group"] = "주재료"

    seasoning_names = {
        "마늘", "소금", "설탕", "간장", "고추장", "된장",
        "식초", "참기름", "들기름", "후추", "고춧가루",
        "올리고당", "맛술", "미림", "케첩", "마요네즈"
    }

    name_column = (
        "ingredient_normalized"
        if "ingredient_normalized"
        in st.session_state.fridge_df.columns
        else "ingredient_input"
    )

    seasoning_mask = (
        st.session_state.fridge_df[name_column]
        .astype(str)
        .isin(seasoning_names)
    )

    st.session_state.fridge_df.loc[
        seasoning_mask,
        "ingredient_group"
    ] = "양념·조미료"

    st.session_state.fridge_df.to_pickle(INVENTORY_PATH)



# 비어 있는 재료 분류 보정
if "ingredient_group" not in st.session_state.fridge_df.columns:
    st.session_state.fridge_df["ingredient_group"] = "주재료"

st.session_state.fridge_df["ingredient_group"] = (
    st.session_state.fridge_df["ingredient_group"]
    .fillna("주재료")
    .replace("", "주재료")
)

st.session_state.fridge_df.to_pickle(INVENTORY_PATH)


# 피드백 로그 불러오기
interaction_columns = [
    "event_id",
    "event_time",
    "RCP_SEQ",
    "RCP_NM",
    "action",
    "preference_label",
    "recommended_rank",
    "matched_ingredients",
    "available_ingredients",
    "hybrid_score",
    "expiry_score",
    "final_score"
]

if "feedback_log_df" not in st.session_state:
    if FEEDBACK_PATH.exists():
        st.session_state.feedback_log_df = pd.read_pickle(
            FEEDBACK_PATH
        )
    else:
        st.session_state.feedback_log_df = pd.DataFrame(
            columns=interaction_columns
        )

st.title("🍳 냉장고를 부탁해")
st.caption("보유 재료와 소비기한을 고려해 활용하기 좋은 레시피를 추천합니다.")

st.subheader("🧊 내 냉장고")

fridge_df = st.session_state.fridge_df

display_columns = [
    "ingredient_input",
    "ingredient_group",
    "quantity",
    "unit",
    "expiry_date",
    "date_type",
    "storage",
    "expiry_status"
]

display_columns = [
    column for column in display_columns
    if column in fridge_df.columns
]

fridge_view = fridge_df[display_columns].copy()

fridge_view = fridge_view.rename(
    columns={
        "ingredient_input": "재료",
        "ingredient_group": "분류",
        "quantity": "수량",
        "unit": "단위",
        "expiry_date": "기한",
        "date_type": "기한 종류",
        "storage": "보관 방법",
        "expiry_status": "기한 상태"
    }
)

st.dataframe(
    fridge_view,
    use_container_width=True,
    hide_index=True
)

st.info(f"현재 냉장고에 {len(fridge_df)}개의 재료가 등록되어 있습니다.")



# --- 냉장고 재료 관리 ---
st.subheader("✏️ 냉장고 재료 관리")

# 기존 데이터에 fridge_id가 없을 경우 생성
if "fridge_id" not in st.session_state.fridge_df.columns:
    st.session_state.fridge_df["fridge_id"] = range(
        1,
        len(st.session_state.fridge_df) + 1
    )
    st.session_state.fridge_df.to_pickle(INVENTORY_PATH)

manage_df = st.session_state.fridge_df

if manage_df.empty:
    st.info("수정하거나 삭제할 재료가 없습니다.")

else:
    item_ids = manage_df["fridge_id"].tolist()

    def fridge_item_label(fridge_id):
        selected = manage_df[
            manage_df["fridge_id"] == fridge_id
        ].iloc[0]

        return (
            f"{selected['ingredient_input']} "
            f"({selected['quantity']}{selected['unit']})"
        )

    edit_tab, delete_tab = st.tabs(["재료 수정", "재료 삭제"])

    # 재료 수정
    with edit_tab:
        edit_id = st.selectbox(
            "수정할 재료",
            options=item_ids,
            format_func=fridge_item_label,
            key="edit_fridge_id"
        )

        edit_row = manage_df[
            manage_df["fridge_id"] == edit_id
        ].iloc[0]

        unit_options = ["개", "g", "kg", "ml", "L", "팩", "모"]
        current_unit = str(edit_row.get("unit", "개"))

        if current_unit not in unit_options:
            unit_options.insert(0, current_unit)

        expiry_value = pd.to_datetime(
            edit_row.get("expiry_date"),
            errors="coerce"
        )

        if pd.isna(expiry_value):
            expiry_value = (
                pd.Timestamp.today() + pd.Timedelta(days=7)
            )

        with st.form("edit_fridge_item_form"):
            edit_col1, edit_col2, edit_col3 = st.columns(3)

            with edit_col1:
                edited_ingredient = st.text_input(
                    "재료명",
                    value=str(edit_row["ingredient_input"])
                )

                edited_quantity = st.number_input(
                    "수량",
                    min_value=0.1,
                    value=float(edit_row["quantity"]),
                    step=1.0
                )

            with edit_col2:
                edited_unit = st.selectbox(
                    "단위",
                    unit_options,
                    index=unit_options.index(current_unit)
                )

                edited_expiry_date = st.date_input(
                    "기한",
                    value=expiry_value.date()
                )

            with edit_col3:
                date_type_options = [
                    "소비기한",
                    "유통기한",
                    "직접입력"
                ]

                current_date_type = str(
                    edit_row.get("date_type", "소비기한")
                )

                if current_date_type not in date_type_options:
                    date_type_options.insert(0, current_date_type)

                edited_date_type = st.selectbox(
                    "기한 종류",
                    date_type_options,
                    index=date_type_options.index(
                        current_date_type
                    )
                )

                storage_options = ["냉장", "냉동", "실온"]
                current_storage = str(
                    edit_row.get("storage", "냉장")
                )

                if current_storage not in storage_options:
                    storage_options.insert(0, current_storage)

                edited_storage = st.selectbox(
                    "보관 방법",
                    storage_options,
                    index=storage_options.index(
                        current_storage
                    )
                )

                group_options = [
                    "주재료",
                    "양념·조미료"
                ]

                current_group = str(
                    edit_row.get(
                        "ingredient_group",
                        "주재료"
                    )
                )

                if current_group not in group_options:
                    current_group = "주재료"

                edited_ingredient_group = st.selectbox(
                    "재료 분류",
                    group_options,
                    index=group_options.index(
                        current_group
                    )
                )

            edit_button = st.form_submit_button(
                "수정 내용 저장",
                use_container_width=True
            )

        if edit_button:
            if not edited_ingredient.strip():
                st.warning("재료명을 입력해주세요.")

            else:
                # 기존 항목을 제외하고 수정된 값으로 다시 등록
                remaining_df = manage_df[
                    manage_df["fridge_id"] != edit_id
                ].copy()

                updated_df = engine["add_fridge_item"](
                    inventory_df=remaining_df,
                    ingredient=edited_ingredient.strip(),
                    quantity=edited_quantity,
                    unit=edited_unit,
                    expiry_date=pd.Timestamp(
                        edited_expiry_date
                    ),
                    date_type=edited_date_type,
                    storage=edited_storage
                )

                if "ingredient_group" not in updated_df.columns:
                    updated_df["ingredient_group"] = "주재료"

                updated_df.loc[
                    updated_df.index[-1],
                    "ingredient_group"
                ] = edited_ingredient_group

                st.session_state.fridge_df = updated_df
                updated_df.to_pickle(INVENTORY_PATH)

                # 이전 추천 결과 제거
                st.session_state.pop(
                    "recommendation_result",
                    None
                )

                st.success("재료 정보를 수정했습니다.")
                st.rerun()

    # 재료 삭제
    with delete_tab:
        delete_id = st.selectbox(
            "삭제할 재료",
            options=item_ids,
            format_func=fridge_item_label,
            key="delete_fridge_id"
        )

        delete_name = manage_df.loc[
            manage_df["fridge_id"] == delete_id,
            "ingredient_input"
        ].iloc[0]

        st.warning(
            f"'{delete_name}'을 냉장고에서 삭제합니다."
        )

        if st.button(
            "선택한 재료 삭제",
            type="primary",
            use_container_width=True
        ):
            updated_df = manage_df[
                manage_df["fridge_id"] != delete_id
            ].copy()

            st.session_state.fridge_df = updated_df
            updated_df.to_pickle(INVENTORY_PATH)

            st.session_state.pop(
                "recommendation_result",
                None
            )

            st.success(f"'{delete_name}'을 삭제했습니다.")
            st.rerun()


# --- 냉장고 재료 추가 폼 ---
from datetime import date, timedelta

st.subheader("➕ 냉장고 재료 추가")

with st.form("add_fridge_item_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        new_ingredient = st.text_input(
            "재료명",
            placeholder="예: 감자"
        )

        new_quantity = st.number_input(
            "수량",
            min_value=0.1,
            value=1.0,
            step=1.0
        )

    with col2:
        new_unit = st.selectbox(
            "단위",
            ["개", "g", "kg", "ml", "L", "팩", "모"]
        )

        new_expiry_date = st.date_input(
            "기한",
            value=date.today() + timedelta(days=7)
        )

    with col3:
        new_date_type = st.selectbox(
            "기한 종류",
            ["소비기한", "유통기한", "직접입력"]
        )

        new_storage = st.selectbox(
            "보관 방법",
            ["냉장", "냉동", "실온"]
        )

        new_ingredient_group = st.selectbox(
            "재료 분류",
            ["주재료", "양념·조미료"]
        )

    add_button = st.form_submit_button(
        "냉장고에 추가",
        use_container_width=True
    )

if add_button:
    if not new_ingredient.strip():
        st.warning("재료명을 입력해주세요.")
    else:
        updated_fridge = engine["add_fridge_item"](
            inventory_df=st.session_state.fridge_df,
            ingredient=new_ingredient.strip(),
            quantity=new_quantity,
            unit=new_unit,
            expiry_date=pd.Timestamp(new_expiry_date),
            date_type=new_date_type,
            storage=new_storage
        )

        if "ingredient_group" not in updated_fridge.columns:
            updated_fridge["ingredient_group"] = "주재료"

        updated_fridge.loc[
            updated_fridge.index[-1],
            "ingredient_group"
        ] = new_ingredient_group

        st.session_state.fridge_df = updated_fridge
        updated_fridge.to_pickle(INVENTORY_PATH)

        st.success(f"'{new_ingredient}'을 냉장고에 추가했습니다.")
        st.rerun()



# --- 피드백 기반 개인화 재정렬 ---
def apply_feedback_reranking(
    recommendation_df,
    feedback_df,
    preference_weight=10
):
    result = recommendation_df.copy()

    # 추천 결과가 없거나 점수 열이 없으면 그대로 반환
    if result.empty or "final_score" not in result.columns:
        return result

    result["preference_adjustment"] = 0.0
    result["personalized_score"] = result["final_score"]

    if feedback_df is None or feedback_df.empty:
        return result

    valid_feedback = feedback_df.copy()

    valid_feedback["preference_label"] = pd.to_numeric(
        valid_feedback["preference_label"],
        errors="coerce"
    )

    valid_feedback = valid_feedback.dropna(
        subset=["RCP_SEQ", "preference_label"]
    )

    if valid_feedback.empty:
        return result

    preference_summary = (
        valid_feedback
        .groupby("RCP_SEQ")["preference_label"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # 좋아요는 양수, 관심 없음은 음수로 계산
    preference_summary["preference_adjustment"] = (
        (preference_summary["mean"] * 2 - 1)
        * preference_weight
        * (
            preference_summary["count"]
            / (preference_summary["count"] + 1)
        )
    )

    result = result.merge(
        preference_summary[
            ["RCP_SEQ", "preference_adjustment"]
        ],
        on="RCP_SEQ",
        how="left",
        suffixes=("", "_feedback")
    )

    if "preference_adjustment_feedback" in result.columns:
        result["preference_adjustment"] = (
            result["preference_adjustment_feedback"]
            .fillna(0)
        )

        result = result.drop(
            columns=["preference_adjustment_feedback"]
        )

    result["personalized_score"] = (
        result["final_score"]
        + result["preference_adjustment"]
    )

    return (
        result
        .sort_values(
            "personalized_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

# --- 레시피 추천 결과 ---
st.divider()
st.subheader("🍽️ 냉장고 재료로 레시피 추천")

selected_category = st.selectbox(
    "음식 종류",
    [
        "전체",
        "반찬",
        "일품",
        "밥",
        "국&찌개",
        "후식",
        "기타"
    ]
)

with st.expander("⚙️ 세부 조건 설정 · 선택사항"):
    selected_method = st.selectbox(
        "조리 방법",
        [
            "전체",
            "끓이기",
            "굽기",
            "볶기",
            "찌기",
            "튀기기",
            "기타"
        ]
    )

    excluded_text = st.text_input(
        "제외할 재료",
        placeholder="예: 돼지고기, 새우"
    )

    st.markdown("##### 영양 조건")

    nutrition_col1, nutrition_col2, nutrition_col3 = (
        st.columns(3)
    )

    with nutrition_col1:
        use_kcal = st.checkbox("칼로리 제한")
        max_kcal_value = st.number_input(
            "최대 kcal",
            min_value=50,
            max_value=2000,
            value=500,
            step=50
        )

    with nutrition_col2:
        use_sodium = st.checkbox("나트륨 제한")
        max_sodium_value = st.number_input(
            "최대 나트륨(mg)",
            min_value=0,
            max_value=5000,
            value=500,
            step=50
        )

    with nutrition_col3:
        use_protein = st.checkbox("최소 단백질")
        min_protein_value = st.number_input(
            "최소 단백질(g)",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

recommend_button = st.button(
    "레시피 5개 추천받기",
    type="primary",
    use_container_width=True
)

if recommend_button:
    excluded_ingredients = [
        ingredient.strip()
        for ingredient in excluded_text.split(",")
        if ingredient.strip()
    ]

    category = (
        None
        if selected_category == "전체"
        else selected_category
    )

    cooking_method = (
        None
        if selected_method == "전체"
        else selected_method
    )

    try:
        recommendation_result = engine["recommend_final"](
            inventory_df=st.session_state.fridge_df,
            excluded_ingredients=excluded_ingredients,
            top_n=5,
            max_missing=5,
            max_kcal=(
                max_kcal_value if use_kcal else None
            ),
            max_sodium=(
                max_sodium_value if use_sodium else None
            ),
            min_protein=(
                min_protein_value if use_protein else None
            ),
            cooking_method=cooking_method,
            category=category
        )

        recommendation_result = (
            apply_feedback_reranking(
                recommendation_result,
                st.session_state.feedback_log_df
            )
        )

        st.session_state.recommendation_result = (
            recommendation_result
        )

    except Exception as error:
        st.error(f"추천 중 오류가 발생했습니다: {error}")


def format_ingredient_list(value):
    if isinstance(value, list):
        return ", ".join(map(str, value))

    if value is None:
        return "-"

    return str(value)


if "recommendation_result" in st.session_state:
    result = st.session_state.recommendation_result

    if result.empty:
        st.warning("조건에 맞는 추천 레시피가 없습니다.")

    else:
        st.success(f"{len(result)}개의 레시피를 추천합니다.")

        for rank, (_, recipe) in enumerate(result.iterrows(), start=1):
            with st.container(border=True):
                image_column, detail_column = st.columns([1, 2])

                with image_column:
                    image_url = recipe.get("ATT_FILE_NO_MAIN")

                    if pd.notna(image_url) and str(image_url).strip():
                        st.image(str(image_url), width=260)

                with detail_column:
                    st.markdown(
                        f"### {rank}. {recipe.get('RCP_NM', '레시피')}"
                    )

                    st.write(
                        f"**활용 가능한 재료:** "
                        f"{format_ingredient_list(recipe.get('matched_ingredients'))}"
                    )

                    st.write(
                        f"**추가로 필요한 재료:** "
                        f"{format_ingredient_list(recipe.get('missing_ingredients'))}"
                    )

                    personalized_score = recipe.get(
                        "personalized_score",
                        recipe.get("final_score", "-")
                    )

                    try:
                        score_text = (
                            f"{float(personalized_score):.1f}"
                        )
                    except (TypeError, ValueError):
                        score_text = str(personalized_score)

                    st.write(
                        f"**개인화 추천 점수:** {score_text}"
                    )

                    adjustment = recipe.get(
                        "preference_adjustment",
                        0
                    )

                    if pd.notna(adjustment) and abs(
                        float(adjustment)
                    ) > 0.01:
                        st.caption(
                            "사용자 피드백 반영: "
                            f"{float(adjustment):+.1f}점"
                        )

                    reason = recipe.get("recommendation_reason")

                    if pd.notna(reason):
                        st.info(str(reason))

                    recipe_name = str(
                        recipe.get("RCP_NM", "요리")
                    )

                    youtube_query = quote_plus(recipe_name)

                    youtube_url = (
                        "https://www.youtube.com/results"
                        f"?search_query={youtube_query}"
                    )

                    naver_query = quote_plus(
                        recipe_name
                    )

                    naver_url = (
                        "https://search.naver.com/"
                        "search.naver"
                        f"?query={naver_query}"
                    )

                    youtube_column, naver_column = (
                        st.columns(2)
                    )

                    with youtube_column:
                        st.link_button(
                            "▶️ 유튜브 검색",
                            youtube_url,
                            use_container_width=True
                        )

                    with naver_column:
                        st.link_button(
                            "🔎 네이버 검색",
                            naver_url,
                            use_container_width=True
                        )

                    like_column, dislike_column = st.columns(2)

                    recipe_id = recipe.get(
                        "RCP_SEQ",
                        rank
                    )

                    with like_column:
                        like_clicked = st.button(
                            "👍 좋아요",
                            key=f"like_{recipe_id}_{rank}",
                            use_container_width=True
                        )

                    with dislike_column:
                        dislike_clicked = st.button(
                            "👎 관심 없음",
                            key=f"dislike_{recipe_id}_{rank}",
                            use_container_width=True
                        )

                    if like_clicked or dislike_clicked:
                        selected_action = (
                            "좋아요"
                            if like_clicked
                            else "싫어요"
                        )

                        updated_log = engine[
                            "log_recipe_action"
                        ](
                            log_df=st.session_state.feedback_log_df,
                            recipe=recipe,
                            action=selected_action,
                            inventory_df=st.session_state.fridge_df,
                            recommended_rank=rank
                        )

                        st.session_state.feedback_log_df = (
                            updated_log
                        )

                        updated_log.to_pickle(
                            FEEDBACK_PATH
                        )

                        if like_clicked:
                            st.success("좋아요를 기록했습니다.")
                        else:
                            st.info("관심 없음으로 기록했습니다.")
