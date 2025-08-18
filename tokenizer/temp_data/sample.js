
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

class ArrayUtils {
    static sum(arr) {
        return arr.reduce((a, b) => a + b, 0);
    }
    
    static filter(arr, predicate) {
        return arr.filter(predicate);
    }
}
