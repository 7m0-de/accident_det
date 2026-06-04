## POINTERS IN C++
### 1. Preface (Idea of Pointers)
Memory = collection of cells (each has address)

Pointers store addresses of variables

Used for:
- Efficiency
- Handling large data
- Dynamic memory

### 2. Pointer Definition
`type *variable_name;`
#### Examples:
```
int *height;
char *age;
```
### 3. Benefits of Pointers
- Save memory
- Reduce program complexity
- Faster execution
- Efficient for arrays & strings
- Allow multiple values return (indirectly)

### 4. Basic Pointer Example
```
int n = 20;
int *ptr;

ptr = &n;

cout << &n;   // address of n
cout << ptr;  // same address
cout << *ptr; // value (20)
```
### 5. Memory Concept
Variable:
```
int x = 10;
int *p = &x;
```
Meaning:
```
p → address of x
*p → value of x
```

### 6. Pointer Syntax
```
int *p;
double *d;
char *c;
```
### 7. Dereferencing (*)
`*p = value at address`

Example:
```
int x = 10;
int *p = &x;

cout << *p; // 10
```
### 8. Modify Value Using Pointer
```
int x = 10;
int *p = &x;

*p = 20;

cout << x; // 20
```

### 9. NULL Pointer
`int *p = NULL;`
→ Points to nothing

### 10. Pointers & Arrays
```
int arr[3] = {10, 20, 30};
int *p = arr;
```
- arr = address of first element
- Access:

```
p[0] = 10;
p[1] = 20;
```
#### Loop:
```
for(int i = 0; i < 3; i++) {
    cout << *(p + i);
}
```
### 11. Pointer Arithmetic
```
cout << *p;     // 10
cout << *(p+1); // 20
cout << *(p+2); // 30
```
### 12. Pointers with Functions
❌ Pass by Value
```
void change(int a) {
    a = 50;
}
```
##### Pass by Pointer
```
void change(int *a) {
    *a = 50;
}
```
Call:
`change(&x);`

### 13. Swapping (Important)
❌ Wrong
`void swap(int a, int b)`

Correct
```
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}
```

### 14. Returning Pointer
```
int* getPointer(int* a) {
    return a;
}
```
### 15. Practice Function
```
void increment(int* num) {
    (*num)++;
}
```

### 16. Pointer to Pointer
```
int x = 10;
int *p = &x;
int **pp = &p;

cout << **pp; // 10
```
### 17. Dynamic Memory (Very Important)
Single value
```
int *p = new int;
*p = 50;

delete p;
```
Array
```
int *arr = new int[3];

delete[] arr;
```
### 18. Dangling Pointer
```
int *p = new int(10);
delete p;
```
Fix:
`p = nullptr;`

### 19. Common Mistakes

❌ Uninitialized pointer:

```
int *p;
*p = 10; // ERROR
```
❌ Forget delete → memory leak

### 20. Real-Life Uses
- Dynamic memory
- Data structures (linked list, tree)
- Efficient data passing
- System-level programming

---
### 21. Mini Problems
**Q1**
```
int x = 5;
int *p = &x;
*p = *p + 10;
```
Output: 15

**Q2**
```
int arr[] = {1,2,3};
int *p = arr;

cout << *(p+2);
```
Output: 3

**Q3**
```
int x = 10;
int *p = &x;
int **pp = &p;

**pp = 50;
```
Output: 50
---
**Pointers** let functions modify the original variable, not just a copy.
