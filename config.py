answer_examples = [
    {
        "input": "SELECT * FROM customers c WHERE c.customer_id IN (SELECT o.customer_id FROM orders o WHERE o.order_total > 1000);",
        "answer": """① IN 절이 포함된 서브쿼리는 서브쿼리 결과가 많을 경우, 옵티마이저에 의해 비효율적인 Nested Loops Join 또는 Hash Semi Join으로 처리될 수 있으며, 특히 서브쿼리 안쪽 테이블에 인덱스가 없을 경우 성능 저하가 발생할 수 있습니다.

② Oracle 공식 문서(19c SQL Tuning Guide > "Choosing Between EXISTS and IN")에 따르면, IN 절보다 EXISTS 절이 옵티마이저에 더 많은 실행 계획 선택의 자유를 줄 수 있습니다.

③ 개선된 쿼리 예시:
   SELECT *
   FROM customers c
   WHERE EXISTS (
       SELECT 1
       FROM orders o
       WHERE o.customer_id = c.customer_id
         AND o.order_total > 1000
   );

④ 실무 팁: EXISTS는 조건이 참이 되는 첫 행만 찾으면 되므로, 대량 서브쿼리에서는 EXISTS가 일반적으로 더 빠릅니다. 특히 서브쿼리에 인덱스가 있는 경우 성능 차이가 큽니다."""
    }
]