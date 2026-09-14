
-- 1. MONTHLY REVENUE TREND (delivered orders only - undelivered orders
--    haven't realized revenue and would distort the trend)
SELECT
    strftime('%Y-%m', o.order_purchase_timestamp) AS year_month,
    COUNT(DISTINCT o.order_id)                     AS num_orders,
    ROUND(SUM(p.payment_value), 2)                 AS revenue,
    ROUND(AVG(p.payment_value), 2)                 AS avg_order_value
FROM orders o
JOIN payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY year_month
ORDER BY year_month;

-- 2. TOP 10 PRODUCT CATEGORIES BY REVENUE
SELECT
    COALESCE(t.product_category_name_english, pr.product_category_name) AS category,
    COUNT(DISTINCT oi.order_id)   AS num_orders,
    ROUND(SUM(oi.price), 2)       AS revenue,
    ROUND(AVG(oi.price), 2)       AS avg_item_price
FROM order_items oi
JOIN products pr ON oi.product_id = pr.product_id
LEFT JOIN product_category_name_translation t
    ON pr.product_category_name = t.product_category_name
JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY category
ORDER BY revenue DESC
LIMIT 10;

-- 3. REVENUE AND AOV BY CUSTOMER STATE
SELECT
    c.customer_state,
    COUNT(DISTINCT o.order_id)   AS num_orders,
    ROUND(SUM(p.payment_value), 2) AS revenue,
    ROUND(AVG(p.payment_value), 2) AS avg_order_value
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN payments p ON o.order_id = p.order_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY revenue DESC;

-- 4. DELIVERY PERFORMANCE: on-time vs late rate, average delay
SELECT
    CASE WHEN julianday(order_delivered_customer_date) > julianday(order_estimated_delivery_date)
         THEN 'Late' ELSE 'On-time/Early' END AS delivery_status,
    COUNT(*) AS num_orders,
    ROUND(AVG(julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp)), 1) AS avg_delivery_days
FROM orders
WHERE order_status = 'delivered' AND order_delivered_customer_date IS NOT NULL
GROUP BY delivery_status;

-- 5. DOES LATE DELIVERY HURT REVIEW SCORES? (avg review score by delivery status)
SELECT
    CASE WHEN julianday(o.order_delivered_customer_date) > julianday(o.order_estimated_delivery_date)
         THEN 'Late' ELSE 'On-time/Early' END AS delivery_status,
    COUNT(*)                        AS num_orders,
    ROUND(AVG(r.review_score), 2)   AS avg_review_score
FROM orders o
JOIN reviews r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date IS NOT NULL
GROUP BY delivery_status;

-- 6. PAYMENT METHOD BREAKDOWN
SELECT
    payment_type,
    COUNT(DISTINCT order_id)       AS num_orders,
    ROUND(SUM(payment_value), 2)   AS total_value,
    ROUND(AVG(payment_installments), 1) AS avg_installments
FROM payments
GROUP BY payment_type
ORDER BY total_value DESC;

-- 7. TOP 10 SELLERS BY REVENUE (a simple seller leaderboard for ops/BI use)
SELECT
    s.seller_id,
    s.seller_state,
    COUNT(DISTINCT oi.order_id)   AS num_orders,
    ROUND(SUM(oi.price), 2)       AS revenue,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM order_items oi
JOIN sellers s ON oi.seller_id = s.seller_id
JOIN orders o ON oi.order_id = o.order_id AND o.order_status = 'delivered'
LEFT JOIN reviews r ON oi.order_id = r.order_id
GROUP BY s.seller_id
ORDER BY revenue DESC
LIMIT 10;

-- 8. REPEAT CUSTOMER RATE (customer_unique_id, since customer_id is
--    generated fresh per order in this dataset - a well-known Olist quirk)
SELECT
    CASE WHEN order_count > 1 THEN 'Repeat customer' ELSE 'One-time customer' END AS customer_type,
    COUNT(*) AS num_customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(DISTINCT customer_unique_id) FROM customers), 2) AS pct_of_customers
FROM (
    SELECT c.customer_unique_id, COUNT(DISTINCT o.order_id) AS order_count
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id
)
GROUP BY customer_type;

-- 9. ORDER STATUS FUNNEL / CANCELLATION RATE
SELECT
    order_status,
    COUNT(*) AS num_orders,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 2) AS pct_of_all_orders
FROM orders
GROUP BY order_status
ORDER BY num_orders DESC;

-- 10. AVERAGE REVIEW SCORE BY PRODUCT CATEGORY (worst 10 - where CX needs work)
SELECT
    COALESCE(t.product_category_name_english, pr.product_category_name) AS category,
    COUNT(*)                       AS num_reviews,
    ROUND(AVG(r.review_score), 2)  AS avg_review_score
FROM reviews r
JOIN order_items oi ON r.order_id = oi.order_id
JOIN products pr ON oi.product_id = pr.product_id
LEFT JOIN product_category_name_translation t ON pr.product_category_name = t.product_category_name
GROUP BY category
HAVING num_reviews >= 30
ORDER BY avg_review_score ASC
LIMIT 10;
