
from pathlib import Path
from urllib.parse import quote_plus
import gzip
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "recommendation_data.pkl.gz"
INVENTORY_PATH = BASE_DIR / "fridge_inventory.pkl"
FEEDBACK_PATH = BASE_DIR / "feedback_log.pkl"


INVENTORY_COLUMNS = [
    "fridge_id",
    "ingredient_input",
    "quantity",
    "unit",
    "expiry_date",
    "date_type",
    "storage",
    "inventory_status",
    "ingredient_normalized",
    "days_left",
    "expiry_status",
    "expiry_priority",
    "recommendation_available",
    "ingredient_group",
]


st.set_page_config(
    page_title="냉장고 레시피 추천",
    page_icon="🍳",
    layout="wide"
)


@st.cache_resource
def load_recommendation_data():
    """함수 객체가 없는 안전한 추천 데이터만 불러옵니다."""
    with gzip.open(DATA_PATH, "rb") as file:
        return pickle.load(file)


def empty_inventory():
    """손상된 저장 파일에서도 앱이 시작되도록 빈 냉장고를 생성합니다."""
    return pd.DataFrame(columns=INVENTORY_COLUMNS)


def safe_load_pickle(path, empty_factory, label):
    """0바이트 또는 불완전한 pickle 파일을 빈 데이터로 복구합니다."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            raise EOFError("empty pickle file")

        loaded = pd.read_pickle(path)

        if not isinstance(loaded, pd.DataFrame):
            raise ValueError("saved object is not a DataFrame")

        return loaded

    except (
        EOFError,
        pickle.UnpicklingError,
        OSError,
        ValueError,
        AttributeError,
    ):
        st.warning(
            f"저장된 {label} 파일이 비어 있거나 손상되어 "
            "빈 상태로 자동 복구했습니다."
        )
        return empty_factory()


def safe_save_pickle(dataframe, path):
    """임시 파일에 먼저 쓴 뒤 교체하여 저장 중 파일 손상을 방지합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        dataframe.to_pickle(temporary_path)
        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_inventory():
    return safe_load_pickle(
        INVENTORY_PATH,
        empty_inventory,
        "냉장고",
    )


recommendation_data = load_recommendation_data()
recipes_df = recommendation_data["recipes"]
tfidf_matrix = recommendation_data["tfidf_matrix"]
tfidf_vocabulary = recommendation_data["vocabulary"]
tfidf_idf = recommendation_data["idf"]
ingredient_map = recommendation_data["ingredient_map"]
basic_seasonings = set(recommendation_data["basic_seasonings"])


def normalize_ingredient(name):
    value = str(name).strip()
    if not value:
        return ""

    if value in ingredient_map:
        return ingredient_map[value]

    compact_value = value.replace(" ", "")
    return ingredient_map.get(compact_value, value)


def refresh_fridge_status(inventory_df):
    result = inventory_df.copy()

    for column in INVENTORY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    if result.empty:
        return result[INVENTORY_COLUMNS]

    result["expiry_date"] = pd.to_datetime(
        result["expiry_date"],
        errors="coerce",
    )
    today = pd.Timestamp.today().normalize()
    result["days_left"] = (
        result["expiry_date"].dt.normalize() - today
    ).dt.days.astype("Int64")

    def expiry_status(days_left):
        if pd.isna(days_left):
            return "미입력"
        if days_left < 0:
            return "기한 지남"
        if days_left <= 2:
            return "임박"
        if days_left <= 7:
            return "곧 임박"
        return "여유"

    def expiry_priority(days_left):
        if pd.isna(days_left) or days_left < 0:
            return 0.0
        if days_left <= 2:
            return 1.0
        if days_left <= 7:
            return 0.5
        return 0.0

    result["expiry_status"] = result["days_left"].apply(
        expiry_status
    )
    result["expiry_priority"] = result["days_left"].apply(
        expiry_priority
    )
    result["ingredient_normalized"] = result[
        "ingredient_input"
    ].apply(normalize_ingredient)
    result["inventory_status"] = result[
        "inventory_status"
    ].fillna("보관중")
    result["recommendation_available"] = (
        result["inventory_status"].eq("보관중")
        & result["days_left"].fillna(0).ge(0)
    )

    return result[INVENTORY_COLUMNS]


def add_fridge_item(
    inventory_df,
    ingredient,
    quantity,
    unit,
    expiry_date=None,
    date_type="소비기한",
    storage="냉장",
):
    if not str(ingredient).strip():
        raise ValueError("재료명을 입력해주세요.")

    current = refresh_fridge_status(inventory_df)
    existing_ids = pd.to_numeric(
        current["fridge_id"],
        errors="coerce",
    ).dropna()
    new_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1

    new_item = pd.DataFrame(
        [{
            "fridge_id": new_id,
            "ingredient_input": str(ingredient).strip(),
            "quantity": float(quantity),
            "unit": unit,
            "expiry_date": pd.to_datetime(expiry_date),
            "date_type": date_type,
            "storage": storage,
            "inventory_status": "보관중",
            "ingredient_normalized": normalize_ingredient(ingredient),
            "days_left": pd.NA,
            "expiry_status": pd.NA,
            "expiry_priority": 0.0,
            "recommendation_available": True,
            "ingredient_group": "주재료",
        }]
    )

    return refresh_fridge_status(
        pd.concat([current, new_item], ignore_index=True)
    )


def make_query_vector(ingredients):
    vector = np.zeros(len(tfidf_vocabulary), dtype=np.float32)

    for ingredient in ingredients:
        feature_index = tfidf_vocabulary.get(ingredient)
        if feature_index is not None:
            vector[feature_index] += 1.0

    vector *= tfidf_idf
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def build_recommendation_reason(
    row,
    max_kcal=None,
    max_sodium=None,
    min_protein=None,
    cooking_method=None,
    category=None,
):
    reasons = []
    expiring = row.get("expiring_ingredients", [])
    matched = row.get("matched_ingredients", [])
    missing_count = int(row.get("missing_count", 0))

    if expiring:
        reasons.append(
            "소비기한이 임박한 재료를 활용할 수 있어요: "
            + ", ".join(expiring)
            + "."
        )

    if missing_count == 0:
        reasons.append("주요 재료를 모두 보유하고 있어 바로 만들기 좋아요.")
    else:
        reasons.append(
            f"보유 재료 {len(matched)}개를 활용할 수 있고 "
            f"추가 재료 {missing_count}개가 필요해요."
        )

    nutrition_conditions = []
    if max_kcal is not None:
        nutrition_conditions.append(f"{max_kcal}kcal 이하")
    if max_sodium is not None:
        nutrition_conditions.append(f"나트륨 {max_sodium}mg 이하")
    if min_protein is not None:
        nutrition_conditions.append(f"단백질 {min_protein}g 이상")
    if nutrition_conditions:
        reasons.append(
            "선택한 영양 조건을 만족해요: "
            + ", ".join(nutrition_conditions)
            + "."
        )
    if category is not None:
        reasons.append(f"선택한 음식 종류인 '{category}'에 해당해요.")
    if cooking_method is not None:
        reasons.append(f"선택한 조리 방법인 '{cooking_method}'에 해당해요.")

    return " ".join(reasons)


def recommend_final(
    inventory_df,
    excluded_ingredients=None,
    top_n=5,
    max_missing=5,
    max_kcal=None,
    max_sodium=None,
    min_protein=None,
    cooking_method=None,
    category=None,
    expiry_weight=0.15,
):
    inventory = refresh_fridge_status(inventory_df)
    available = inventory[
        inventory["recommendation_available"].fillna(False)
    ].copy()

    user_ingredients = list(dict.fromkeys(
        available["ingredient_normalized"]
        .dropna()
        .astype(str)
        .tolist()
    ))
    if not user_ingredients:
        raise ValueError("추천에 사용할 수 있는 냉장고 재료가 없습니다.")

    candidates = recipes_df.copy()
    if category is not None:
        candidates = candidates[candidates["RCP_PAT2"].eq(category)]
    if cooking_method is not None:
        candidates = candidates[candidates["RCP_WAY2"].eq(cooking_method)]
    if max_kcal is not None:
        candidates = candidates[
            pd.to_numeric(candidates["INFO_ENG_CLEAN"], errors="coerce")
            .le(max_kcal)
        ]
    if max_sodium is not None:
        candidates = candidates[
            pd.to_numeric(candidates["INFO_NA_CLEAN"], errors="coerce")
            .le(max_sodium)
        ]
    if min_protein is not None:
        candidates = candidates[
            pd.to_numeric(candidates["INFO_PRO_CLEAN"], errors="coerce")
            .ge(min_protein)
        ]

    excluded = {
        normalize_ingredient(value)
        for value in (excluded_ingredients or [])
        if str(value).strip()
    }
    user_set = set(user_ingredients)
    expiry_map = dict(zip(
        available["ingredient_normalized"].astype(str),
        pd.to_numeric(available["expiry_priority"], errors="coerce")
        .fillna(0.0),
    ))

    query_vector = make_query_vector(user_ingredients)
    similarity = tfidf_matrix @ query_vector
    rows = []

    for index, recipe in candidates.iterrows():
        recipe_ingredients = recipe.get("ingredients_normalized", [])
        if not isinstance(recipe_ingredients, list):
            recipe_ingredients = []

        recipe_ingredients = list(dict.fromkeys(
            str(value) for value in recipe_ingredients if str(value).strip()
        ))
        recipe_set = set(recipe_ingredients)

        if excluded and recipe_set.intersection(excluded):
            continue

        matched = [value for value in recipe_ingredients if value in user_set]
        missing = [
            value for value in recipe_ingredients
            if value not in user_set and value not in basic_seasonings
        ]
        if not matched or len(missing) > max_missing:
            continue

        denominator = len(matched) + len(missing)
        match_percent = (
            len(matched) / denominator * 100 if denominator else 0.0
        )
        rule_score = min(100.0, match_percent + len(matched) * 2.5)
        tfidf_score = float(similarity[index]) * 100.0
        hybrid_score = rule_score * 0.7 + tfidf_score * 0.3

        expiring = [
            value for value in matched if expiry_map.get(value, 0.0) > 0
        ]
        expiry_score = (
            float(np.mean([expiry_map[value] for value in expiring])) * 100
            if expiring else 0.0
        )
        final_score = (
            hybrid_score * (1 - expiry_weight)
            + expiry_score * expiry_weight
        )

        output = recipe.to_dict()
        output.update({
            "matched_ingredients": matched,
            "missing_ingredients": missing,
            "missing_count": len(missing),
            "match_percent": round(match_percent, 1),
            "rule_score": round(rule_score, 1),
            "tfidf_score": round(tfidf_score, 1),
            "hybrid_score": round(hybrid_score, 1),
            "expiring_ingredients": expiring,
            "expiry_score": round(expiry_score, 1),
            "final_score": round(final_score, 1),
        })
        output["recommendation_reason"] = build_recommendation_reason(
            output,
            max_kcal=max_kcal,
            max_sodium=max_sodium,
            min_protein=min_protein,
            cooking_method=cooking_method,
            category=category,
        )
        rows.append(output)

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["final_score", "missing_count"],
            ascending=[False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


def log_recipe_action(
    log_df,
    recipe,
    action,
    inventory_df,
    recommended_rank,
):
    current = log_df.copy()
    event_ids = pd.to_numeric(
        current.get("event_id", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    event_id = int(event_ids.max()) + 1 if not event_ids.empty else 1
    preference_label = 1 if action == "좋아요" else 0

    new_log = pd.DataFrame([{
        "event_id": event_id,
        "event_time": pd.Timestamp.now(),
        "RCP_SEQ": recipe.get("RCP_SEQ"),
        "RCP_NM": recipe.get("RCP_NM"),
        "action": action,
        "preference_label": preference_label,
        "recommended_rank": recommended_rank,
        "matched_ingredients": recipe.get("matched_ingredients", []),
        "available_ingredients": inventory_df.get(
            "ingredient_normalized",
            pd.Series(dtype=str),
        ).dropna().astype(str).tolist(),
        "hybrid_score": recipe.get("hybrid_score"),
        "expiry_score": recipe.get("expiry_score"),
        "final_score": recipe.get("final_score"),
    }])

    return pd.concat([current, new_log], ignore_index=True)

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

    safe_save_pickle(st.session_state.fridge_df, INVENTORY_PATH)



# 비어 있는 재료 분류 보정
if "ingredient_group" not in st.session_state.fridge_df.columns:
    st.session_state.fridge_df["ingredient_group"] = "주재료"

st.session_state.fridge_df["ingredient_group"] = (
    st.session_state.fridge_df["ingredient_group"]
    .fillna("주재료")
    .replace("", "주재료")
)

safe_save_pickle(st.session_state.fridge_df, INVENTORY_PATH)


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
    st.session_state.feedback_log_df = safe_load_pickle(
        FEEDBACK_PATH,
        lambda: pd.DataFrame(columns=interaction_columns),
        "사용자 반응",
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
    width="stretch",
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
    safe_save_pickle(st.session_state.fridge_df, INVENTORY_PATH)

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
                width="stretch"
            )

        if edit_button:
            if not edited_ingredient.strip():
                st.warning("재료명을 입력해주세요.")

            else:
                # 기존 항목을 제외하고 수정된 값으로 다시 등록
                remaining_df = manage_df[
                    manage_df["fridge_id"] != edit_id
                ].copy()

                updated_df = add_fridge_item(
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
                safe_save_pickle(updated_df, INVENTORY_PATH)

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
            width="stretch"
        ):
            updated_df = manage_df[
                manage_df["fridge_id"] != delete_id
            ].copy()

            st.session_state.fridge_df = updated_df
            safe_save_pickle(updated_df, INVENTORY_PATH)

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
        width="stretch"
    )

if add_button:
    if not new_ingredient.strip():
        st.warning("재료명을 입력해주세요.")
    else:
        updated_fridge = add_fridge_item(
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
        safe_save_pickle(updated_fridge, INVENTORY_PATH)

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
    width="stretch"
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
        recommendation_result = recommend_final(
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
                            width="stretch"
                        )

                    with naver_column:
                        st.link_button(
                            "🔎 네이버 검색",
                            naver_url,
                            width="stretch"
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
                            width="stretch"
                        )

                    with dislike_column:
                        dislike_clicked = st.button(
                            "👎 관심 없음",
                            key=f"dislike_{recipe_id}_{rank}",
                            width="stretch"
                        )

                    if like_clicked or dislike_clicked:
                        selected_action = (
                            "좋아요"
                            if like_clicked
                            else "싫어요"
                        )

                        updated_log = log_recipe_action(
                            log_df=st.session_state.feedback_log_df,
                            recipe=recipe,
                            action=selected_action,
                            inventory_df=st.session_state.fridge_df,
                            recommended_rank=rank
                        )

                        st.session_state.feedback_log_df = (
                            updated_log
                        )

                        safe_save_pickle(
                            updated_log,
                            FEEDBACK_PATH,
                        )

                        if like_clicked:
                            st.success("좋아요를 기록했습니다.")
                        else:
                            st.info("관심 없음으로 기록했습니다.")
